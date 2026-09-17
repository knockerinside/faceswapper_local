"""Thread-safe event system for decoupled GUI and pipeline communication."""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import threading
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    STATUS_CHANGED = auto()          # "Loading models...", "Ready", etc.
    CAMERA_CONNECTED = auto()
    CAMERA_DISCONNECTED = auto()
    FRAME_PROCESSED = auto()
    SOURCE_FACE_LOADED = auto()
    SOURCE_FACE_ERROR = auto()
    PIPELINE_CRASH = auto()
    ADAPTIVE_CHANGE = auto()
    PROFILER_UPDATE = auto()
    VIRTUAL_CAM_STATUS = auto()


@dataclass
class PipelineEvent:
    event_type: EventType
    data: Any = None
    timestamp: float = 0.0


class EventBus:
    """Thread-safe publish-subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[PipelineEvent], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable[[PipelineEvent], None]) -> None:
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[PipelineEvent], None]) -> None:
        with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def publish(self, event: PipelineEvent) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event.event_type, []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error dispatching event {event.event_type}: {e}", exc_info=True)


# Global singleton event bus
event_bus = EventBus()
