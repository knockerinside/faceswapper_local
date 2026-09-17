"""Bounded low-latency frame buffer (Max capacity: 1 frame).

Design Mandate (Sections 6, 50, 51):
- Never allow frames to queue up behind AI inference.
- If processing is slower than capture (e.g. camera 60fps, AI 35fps), stale frames
  are dropped immediately.
- Buffer capacity is strictly 1 frame.
- Zero busy-waiting: uses thread synchronization (Condition variables) with timeouts.
"""
import time
import threading
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    frame_id: int
    capture_timestamp: float
    width: int
    height: int
    dropped_before_this: int = 0


class LatestFrameBuffer:
    """Thread-safe bounded buffer holding exactly ONE latest frame.
    
    Drops older incoming frames when new ones arrive before extraction.
    """

    def __init__(self, name: str = "CaptureBuffer") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        self._latest_frame: Optional[Any] = None
        self._latest_metadata: Optional[FrameMetadata] = None
        self._has_new_frame: bool = False
        self._is_closed: bool = False

        # Operational Metrics
        self.captured_count: int = 0
        self.dropped_count: int = 0
        self.extracted_count: int = 0
        
        # FPS Tracking
        self._last_capture_time: float = time.perf_counter()
        self._last_extract_time: float = time.perf_counter()
        self._capture_fps_history: list[float] = []
        self._extract_fps_history: list[float] = []

    def put(self, frame: Any, width: int = 0, height: int = 0) -> None:
        """Pushes a new frame into the buffer. If an unread frame exists, it is dropped."""
        now = time.perf_counter()
        with self._lock:
            if self._is_closed:
                return

            self.captured_count += 1

            if self._has_new_frame:
                # Discard the stale frame
                self.dropped_count += 1

            metadata = FrameMetadata(
                frame_id=self.captured_count,
                capture_timestamp=now,
                width=width,
                height=height,
                dropped_before_this=self.dropped_count,
            )

            self._latest_frame = frame
            self._latest_metadata = metadata
            self._has_new_frame = True

            # Notify waiting worker thread
            self._condition.notify()

    def get(self, timeout: Optional[float] = 0.5) -> Tuple[Optional[Any], Optional[FrameMetadata]]:
        """Extracts the latest available frame. Waits up to timeout seconds.
        
        Returns (None, None) if timeout occurs or buffer is closed.
        """
        with self._lock:
            start_wait = time.perf_counter()
            while not self._has_new_frame and not self._is_closed:
                remaining = timeout
                if timeout is not None:
                    elapsed = time.perf_counter() - start_wait
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return None, None
                self._condition.wait(timeout=remaining)

            if self._is_closed or not self._has_new_frame:
                return None, None

            frame = self._latest_frame
            metadata = self._latest_metadata
            self._has_new_frame = False
            self.extracted_count += 1
            return frame, metadata

    def clear(self) -> None:
        with self._lock:
            self._latest_frame = None
            self._latest_metadata = None
            self._has_new_frame = False

    def close(self) -> None:
        with self._lock:
            self._is_closed = True
            self._condition.notify_all()

    def reset_stats(self) -> None:
        with self._lock:
            self.captured_count = 0
            self.dropped_count = 0
            self.extracted_count = 0

    @property
    def drop_rate_percentage(self) -> float:
        with self._lock:
            if self.captured_count == 0:
                return 0.0
            return (self.dropped_count / self.captured_count) * 100.0

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "buffer_name": self.name,
                "captured": self.captured_count,
                "dropped": self.dropped_count,
                "extracted": self.extracted_count,
                "drop_rate_pct": round(self.drop_rate_percentage, 1),
                "is_closed": self._is_closed,
            }
