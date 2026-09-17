"""Master Dual-Thread Face Swap Pipeline with Bounded Buffers and Adaptive Quality.

Requirements (Sections 2, 6, 32, 33, 48, 50, 52, 67):
- Thread 1 (Capture Worker): Reads camera, pushes to bounded buffer (capacity=1).
- Thread 2 (Inference Worker): Pulls newest frame, runs detection/tracking/swap/compositing.
- Zero frame queue accumulation.
- Adaptive performance with hysteresis cooldown.
- Resilient crash recovery without killing the application process.
"""
import threading
import time
import logging
from typing import Optional, Tuple, Any
import numpy as np

from .config import AppConfig
from .frame_buffer import LatestFrameBuffer
from .profiler import PerformanceProfiler
from .events import event_bus, PipelineEvent, EventType
from realtime_faceswap.capture.base import CameraBackend
from realtime_faceswap.capture.opencv_dshow import OpenCVCameraBackend
from realtime_faceswap.detection.scrfd import SCRFDDetector
from realtime_faceswap.tracking.face_tracker import FaceTracker
from realtime_faceswap.alignment.face_alignment import FaceAligner
from realtime_faceswap.swapping.inswapper import InSwapperBackend
from realtime_faceswap.parsing.bisenet import BiSeNetParser
from realtime_faceswap.hair.hair_effects import HairProcessor
from realtime_faceswap.compositing.masks import MaskGenerator
from realtime_faceswap.compositing.blender import FaceBlender
from realtime_faceswap.output.virtual_camera import VirtualCameraBackend

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FaceSwapPipeline:
    """Orchestrates multi-threaded real-time video capture, AI swapping, and virtual output."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.buffer = LatestFrameBuffer(name="LiveCaptureBuffer")
        self.profiler = PerformanceProfiler(window_size=60)

        # Pipeline Subsystems
        self.camera_backend: CameraBackend = OpenCVCameraBackend()
        self.detector = SCRFDDetector(
            confidence_threshold=config.detection.confidence_threshold,
            nms_threshold=config.detection.nms_threshold,
            input_size=tuple(config.detection.input_size),
        )
        self.tracker = FaceTracker(
            bbox_smoothing=config.tracking.bbox_smoothing,
            landmark_smoothing=config.tracking.landmark_smoothing,
        )
        self.aligner = FaceAligner(crop_size=128)
        self.swapper = InSwapperBackend()
        self.parser = BiSeNetParser(input_size=tuple(config.parsing.input_size))
        self.hair_processor = HairProcessor()
        self.mask_generator = MaskGenerator(crop_size=128)
        self.blender = FaceBlender()
        self.virtual_camera = VirtualCameraBackend()

        # State and Threading Controls
        self._is_running: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Preview state
        self._latest_processed_frame: Optional[np.ndarray] = None
        self._latest_preview_lock = threading.Lock()

        # Adaptive quality hysteresis counters
        self._frames_below_target: int = 0
        self._cooldown_counter: int = 0
        self._adaptive_tier: int = 0  # 0=normal, 1=reduced parsing, 2=reduced detection

        # Cached Source Face
        self.source_image: Optional[np.ndarray] = None
        self.source_embedding: Optional[np.ndarray] = None
        self.has_source_face: bool = False

    def initialize_models(self) -> None:
        """Loads models ONCE and performs warmup."""
        event_bus.publish(PipelineEvent(EventType.STATUS_CHANGED, "Loading AI models..."))
        
        # Load Detector
        self.detector.load(
            self.config.detection.model_path, self.config.detection.execution_provider
        )
        
        # Load Swapper
        self.swapper.load(
            self.config.swap.model_path, self.config.swap.execution_provider
        )

        # Optional Parser
        if self.config.parsing.enabled:
            self.parser.load(
                self.config.parsing.model_path, self.config.parsing.execution_provider
            )

        event_bus.publish(PipelineEvent(EventType.STATUS_CHANGED, "Warming up GPU..."))
        self.detector.warmup(2)
        self.swapper.warmup(2)
        if self.config.parsing.enabled:
            self.parser.warmup(1)

        event_bus.publish(PipelineEvent(EventType.STATUS_CHANGED, "AI Pipeline Ready"))

    def set_source_face(self, image_path: str) -> bool:
        """Loads source image, extracts face, aligns, and computes ArcFace embedding."""
        if not HAS_CV2:
            return False

        img = cv2.imread(image_path)
        if img is None:
            event_bus.publish(PipelineEvent(EventType.SOURCE_FACE_ERROR, f"Cannot open image: {image_path}"))
            return False

        # Detect source face
        faces = self.detector.detect(img, max_num=1)
        if not faces:
            event_bus.publish(PipelineEvent(EventType.SOURCE_FACE_ERROR, "No face found in source image!"))
            return False

        source_face = faces[0]
        # Align face to 128x128
        M, _ = self.aligner.get_alignment_matrix(source_face.landmarks)
        aligned_source = self.aligner.crop_face(img, M)

        # Pass to swapper backend
        success = self.swapper.set_source(aligned_source, source_face.embedding)
        if success:
            self.source_image = img
            self.has_source_face = True
            event_bus.publish(PipelineEvent(EventType.SOURCE_FACE_LOADED, aligned_source))
            return True

        return False

    def start(self, device_index: int = 0) -> bool:
        """Starts capture and inference threads."""
        with self._lock:
            if self._is_running:
                return True

            # Open camera device
            success = self.camera_backend.open(
                device_index=device_index,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
            if not success:
                event_bus.publish(PipelineEvent(EventType.STATUS_CHANGED, "Camera Connection Failed"))
                return False

            event_bus.publish(PipelineEvent(EventType.CAMERA_CONNECTED, device_index))

            # Start virtual camera if enabled
            if self.config.output.virtual_camera_enabled:
                self.virtual_camera.start(
                    width=self.config.output.width,
                    height=self.config.output.height,
                    fps=self.config.output.fps,
                )

            self.buffer.reset_stats()
            self.profiler.reset()
            self._is_running = True

            # Spawn dedicated capture thread (Thread 1)
            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="CameraCaptureWorker", daemon=True
            )
            # Spawn dedicated inference thread (Thread 2)
            self._inference_thread = threading.Thread(
                target=self._inference_loop, name="AIInferenceWorker", daemon=True
            )

            self._capture_thread.start()
            self._inference_thread.start()
            event_bus.publish(PipelineEvent(EventType.STATUS_CHANGED, "Streaming Active"))
            logger.info("Pipeline started successfully.")
            return True

    def stop(self) -> None:
        """Gracefully shuts down all worker threads and camera handles."""
        with self._lock:
            if not self._is_running:
                return

            self._is_running = False
            self.buffer.close()

            # Wait for worker threads to terminate cleanly
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=1.5)
            if self._inference_thread and self._inference_thread.is_alive():
                self._inference_thread.join(timeout=1.5)

            self.camera_backend.release()
            self.virtual_camera.stop()
            event_bus.publish(PipelineEvent(EventType.STATUS_CHANGED, "Pipeline Stopped"))
            logger.info("Pipeline stopped cleanly.")

    def _capture_loop(self) -> None:
        """Thread 1: Captures frames at camera speed and places them into bounded buffer."""
        logger.info("Capture worker thread started.")
        while self._is_running:
            self.profiler.start_stage("capture")
            ret, frame = self.camera_backend.read()
            cap_time = self.profiler.end_stage("capture")

            if not ret or frame is None:
                # Camera disconnected or stalled
                logger.warning("Camera frame read failed. Checking connection...")
                event_bus.publish(PipelineEvent(EventType.CAMERA_DISCONNECTED, "Camera Disconnected"))
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            self.profiler.record_capture_event()
            # Push into bounded buffer (size=1)
            self.buffer.put(frame, width=w, height=h)

        logger.info("Capture worker thread exiting.")

    def _inference_loop(self) -> None:
        """Thread 2: Pulls latest frame, runs AI pipeline, and feeds output."""
        logger.info("Inference worker thread started.")
        frame_idx = 0

        while self._is_running:
            try:
                # Wait for next available frame
                frame, meta = self.buffer.get(timeout=0.2)
                if frame is None:
                    continue

                frame_idx += 1
                self.profiler.record_process_event()

                processed_frame = self._process_single_frame(frame, frame_idx)

                with self._latest_preview_lock:
                    self._latest_processed_frame = processed_frame

                # Send to virtual camera if active
                if self.virtual_camera.is_active():
                    self.profiler.start_stage("output")
                    self.virtual_camera.send_frame(processed_frame)
                    self.profiler.end_stage("output")
                    self.profiler.record_output_event()

                # Periodic profiler update (every 10 frames)
                if frame_idx % 10 == 0:
                    summary = self.profiler.get_summary()
                    event_bus.publish(PipelineEvent(EventType.PROFILER_UPDATE, summary))
                    if self.config.adaptive.enabled:
                        self._check_adaptive_performance(summary)

            except Exception as e:
                logger.error(f"Critical error in AI inference loop: {e}", exc_info=True)
                event_bus.publish(PipelineEvent(EventType.PIPELINE_CRASH, str(e)))
                time.sleep(0.5)

        logger.info("Inference worker thread exiting.")

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Executes full detection -> tracking -> alignment -> swap -> masking -> composite."""
        if not self.config.swap.enabled or not self.has_source_face:
            return frame

        self.profiler.start_stage("preprocessing")
        orig_frame = frame.copy()
        self.profiler.end_stage("preprocessing")

        # 1. Detection vs Tracking check
        detect_interval = self.config.detection.interval
        should_detect = (frame_idx % detect_interval == 0) or (self.tracker.confidence < 0.35)

        target_face = None
        if should_detect:
            self.profiler.start_stage("detection")
            faces = self.detector.detect(orig_frame, max_num=1)
            target_face = faces[0] if faces else None
            target_face = self.tracker.update_with_detection(target_face)
            self.profiler.end_stage("detection")
        else:
            self.profiler.start_stage("tracking")
            target_face = self.tracker.step_tracking()
            self.profiler.end_stage("tracking")

        if target_face is None:
            return orig_frame

        # 2. Alignment & Crop (128x128)
        self.profiler.start_stage("alignment")
        M, M_inv = self.aligner.get_alignment_matrix(target_face.landmarks)
        aligned_target = self.aligner.crop_face(orig_frame, M)
        self.profiler.end_stage("alignment")

        # 3. InSwapper Face Swap
        self.profiler.start_stage("face_swap")
        swapped_face_128 = self.swapper.process(aligned_target)
        self.profiler.end_stage("face_swap")

        # 4. Optional Parsing & Hair
        hair_mask = None
        custom_face_mask = None
        if self.config.parsing.enabled:
            self.profiler.start_stage("parsing")
            parsed = self.parser.parse(aligned_target)
            custom_face_mask = parsed.face_mask
            hair_mask = parsed.hair_mask
            self.profiler.end_stage("parsing")

        # 5. Mask Generation
        mask_128 = self.mask_generator.generate_mask(
            custom_mask=custom_face_mask,
            feather_px=self.config.swap.mask_feather,
            erosion_px=self.config.swap.mask_erosion,
            temporal_alpha=self.config.swap.mask_smoothing,
        )

        # Optional hair boundary protection
        if self.config.hair.enabled and hair_mask is not None and self.config.hair.boundary_protection:
            mask_128 = self.hair_processor.protect_hair_boundary(
                (mask_128 * 255).astype(np.uint8), hair_mask
            ).astype(np.float32) / 255.0

        # 6. Compositing & Blending
        self.profiler.start_stage("compositing")
        final_frame = self.blender.blend(
            original_frame=orig_frame,
            aligned_target_face=aligned_target,
            swapped_face_128=swapped_face_128,
            mask_128=mask_128,
            M_inv=M_inv,
            blend_strength=self.config.swap.blend_strength,
            enable_color_correction=self.config.swap.color_correction,
        )

        # 7. Optional Hair Recolor
        if self.config.hair.enabled and self.config.hair.recolor_enabled and hair_mask is not None:
            final_frame = self.hair_processor.apply_hair_effects(
                frame=final_frame,
                hair_mask=hair_mask,
                target_color_rgb=tuple(self.config.hair.color_rgb),
                strength=self.config.hair.strength,
            )
        self.profiler.end_stage("compositing")

        # 8. Optional Debug HUD Overlay (Normal output vs Virtual Camera separation)
        if self.config.debug.enabled:
            final_frame = self._draw_debug_overlay(final_frame, target_face)

        return final_frame

    def _draw_debug_overlay(self, frame: np.ndarray, face: Any) -> np.ndarray:
        """Draws bounding box, landmarks, and latency HUD for local debug preview."""
        if not HAS_CV2:
            return frame
        out = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for pt in face.landmarks:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"Track Conf: {self.tracker.confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return out

    def _check_adaptive_performance(self, summary: Any) -> None:
        """Throttles optional passes if FPS drops below threshold (Section 52)."""
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        target = self.config.adaptive.target_fps
        actual = summary.fps_processing

        if actual > 0 and actual < (target - 5.0):
            self._frames_below_target += 1
            if self._frames_below_target > 5:
                # Step down quality tier
                if self.config.parsing.enabled:
                    self.config.parsing.enabled = False
                    logger.info("Adaptive Engine: Disabled hair/face parsing to recover FPS.")
                    event_bus.publish(PipelineEvent(EventType.ADAPTIVE_CHANGE, "Disabled parsing (FPS recovery)"))
                elif self.config.detection.interval < 8:
                    self.config.detection.interval += 2
                    logger.info(f"Adaptive Engine: Increased detection interval to {self.config.detection.interval}")
                    event_bus.publish(PipelineEvent(EventType.ADAPTIVE_CHANGE, f"Detection interval -> {self.config.detection.interval}"))

                self._frames_below_target = 0
                self._cooldown_counter = self.config.adaptive.cooldown_frames
        else:
            self._frames_below_target = max(0, self._frames_below_target - 1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._latest_preview_lock:
            return self._latest_processed_frame
