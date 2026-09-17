"""Integration tests for the dual-threaded face swap pipeline."""
import unittest
import numpy as np
import time
from realtime_faceswap.core.config import AppConfig
from realtime_faceswap.core.pipeline import FaceSwapPipeline
from realtime_faceswap.capture.video_backend import VideoFileCameraBackend


class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_startup_and_shutdown(self):
        """Verify pipeline starts threads and shuts down cleanly without hanging."""
        cfg = AppConfig()
        pipeline = FaceSwapPipeline(cfg)
        # Use mock synthetic video backend
        pipeline.camera_backend = VideoFileCameraBackend()

        # Start
        started = pipeline.start(device_index=0)
        self.assertTrue(started)
        self.assertTrue(pipeline._is_running)

        # Allow dual threads to run briefly
        time.sleep(0.15)

        # Stop
        pipeline.stop()
        self.assertFalse(pipeline._is_running)


if __name__ == "__main__":
    unittest.main()
