"""Unit tests for LatestFrameBuffer (Section 51: Strict Bounded Buffer size=1)."""
import unittest
import time
import threading
from realtime_faceswap.core.frame_buffer import LatestFrameBuffer


class TestLatestFrameBuffer(unittest.TestCase):
    def setUp(self):
        self.buf = LatestFrameBuffer(name="TestBuffer")

    def tearDown(self):
        self.buf.close()

    def test_single_frame_put_and_get(self):
        """Verify normal single-frame insertion and retrieval."""
        self.buf.put("frame_1", width=1280, height=720)
        frame, meta = self.buf.get(timeout=0.1)
        self.assertEqual(frame, "frame_1")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.width, 1280)
        self.assertEqual(self.buf.captured_count, 1)
        self.assertEqual(self.buf.dropped_count, 0)
        self.assertEqual(self.buf.extracted_count, 1)

    def test_bounded_dropping_of_stale_frames(self):
        """Verify that when 5 frames are put in succession without reading, 4 older frames are dropped."""
        for i in range(1, 6):
            self.buf.put(f"frame_{i}")

        # The extracted frame MUST be the latest frame ("frame_5")
        frame, meta = self.buf.get(timeout=0.1)
        self.assertEqual(frame, "frame_5")
        self.assertEqual(self.buf.captured_count, 5)
        self.assertEqual(self.buf.dropped_count, 4)
        self.assertEqual(self.buf.extracted_count, 1)
        self.assertEqual(self.buf.drop_rate_percentage, 80.0)

    def test_timeout_on_empty_buffer(self):
        """Verify get returns None on empty buffer after timeout without hanging."""
        start = time.perf_counter()
        frame, meta = self.buf.get(timeout=0.05)
        elapsed = time.perf_counter() - start
        self.assertIsNone(frame)
        self.assertIsNone(meta)
        self.assertGreaterEqual(elapsed, 0.04)

    def test_concurrent_producer_consumer(self):
        """Simulate fast camera (60fps) and slower AI (30fps) over 0.2s."""
        consumed = []
        is_running = True

        def producer():
            idx = 0
            while is_running:
                idx += 1
                self.buf.put(f"cam_{idx}")
                time.sleep(0.01)  # ~100 FPS

        def consumer():
            while is_running:
                f, _ = self.buf.get(timeout=0.05)
                if f:
                    consumed.append(f)
                time.sleep(0.03)  # ~33 FPS

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()

        time.sleep(0.25)
        is_running = False
        t_prod.join()
        t_cons.join()

        # Check that drops occurred and latest frame freshness was prioritized
        self.assertGreater(self.buf.dropped_count, 0)
        self.assertGreater(len(consumed), 0)


if __name__ == "__main__":
    unittest.main()
