"""Lightweight Face Tracker with Temporal Smoothing (EMA).

Requirements (Sections 7, 17):
- Exponential Moving Average (EMA) to prevent facial landmark and bounding box jitter.
- Interpolates between full detection intervals (every 3-5 frames) without re-running full SCRFD.
- Detects tracking loss and signals immediate redetection.
- Configurable smoothing parameters (0.0=instant response, 0.9=heavy stabilization).
"""
from typing import Optional, Tuple
import numpy as np
from realtime_faceswap.detection.base import DetectedFace


class FaceTracker:
    """Stabilizes face bounding boxes and 5 facial keypoints across video frames."""

    def __init__(self, bbox_smoothing: float = 0.5, landmark_smoothing: float = 0.5) -> None:
        self.bbox_smoothing = bbox_smoothing
        self.landmark_smoothing = landmark_smoothing

        self._current_face: Optional[DetectedFace] = None
        self._smoothed_bbox: Optional[np.ndarray] = None
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._frames_since_detect: int = 0
        self._tracking_confidence: float = 1.0

    def update_with_detection(self, face: Optional[DetectedFace]) -> Optional[DetectedFace]:
        """Called when SCRFD detector has produced a fresh face result."""
        self._frames_since_detect = 0
        if face is None:
            self._current_face = None
            self._smoothed_bbox = None
            self._smoothed_landmarks = None
            self._tracking_confidence = 0.0
            return None

        raw_bbox = np.array(face.bbox, dtype=np.float32)
        raw_kps = face.landmarks.astype(np.float32)

        if self._smoothed_bbox is None:
            # First frame initial lock
            self._smoothed_bbox = raw_bbox.copy()
            self._smoothed_landmarks = raw_kps.copy()
        else:
            # Exponential Moving Average smoothing: S_t = alpha * S_{t-1} + (1 - alpha) * X_t
            alpha_b = self.bbox_smoothing
            alpha_l = self.landmark_smoothing
            self._smoothed_bbox = alpha_b * self._smoothed_bbox + (1.0 - alpha_b) * raw_bbox
            self._smoothed_landmarks = alpha_l * self._smoothed_landmarks + (1.0 - alpha_l) * raw_kps

        self._tracking_confidence = face.score
        self._current_face = DetectedFace(
            bbox=(
                float(self._smoothed_bbox[0]),
                float(self._smoothed_bbox[1]),
                float(self._smoothed_bbox[2]),
                float(self._smoothed_bbox[3]),
            ),
            score=self._tracking_confidence,
            landmarks=self._smoothed_landmarks.copy(),
            embedding=face.embedding,
            track_id=face.track_id,
        )
        return self._current_face

    def step_tracking(self) -> Optional[DetectedFace]:
        """Called on intermediate frames where full SCRFD is skipped.
        
        Decays tracking confidence slightly until next full detection interval.
        """
        self._frames_since_detect += 1
        if self._current_face is None:
            return None

        # Tracking confidence decays gently over consecutive non-detection frames
        decay_factor = max(0.0, 1.0 - (self._frames_since_detect * 0.05))
        self._tracking_confidence = self._current_face.score * decay_factor

        if self._tracking_confidence < 0.35:
            # Confidence lost: trigger fresh detection
            return None

        return self._current_face

    def reset(self) -> None:
        self._current_face = None
        self._smoothed_bbox = None
        self._smoothed_landmarks = None
        self._frames_since_detect = 0
        self._tracking_confidence = 0.0

    @property
    def confidence(self) -> float:
        return self._tracking_confidence

    @property
    def frames_since_detect(self) -> int:
        return self._frames_since_detect
