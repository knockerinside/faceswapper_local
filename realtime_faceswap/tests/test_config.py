"""Unit tests for Configuration and Optimization Profiles (Section 65 & 82)."""
import unittest
from realtime_faceswap.core.config import AppConfig, ProfileType, ConfigManager


class TestConfig(unittest.TestCase):
    def test_safe_defaults(self):
        """Verify Section 82: Safe defaults on initial launch."""
        cfg = AppConfig()
        self.assertEqual(cfg.camera.width, 1280)
        self.assertEqual(cfg.camera.height, 720)
        self.assertEqual(cfg.camera.fps, 30)
        self.assertTrue(cfg.swap.enabled)
        self.assertFalse(cfg.parsing.enabled)
        self.assertFalse(cfg.hair.enabled)
        self.assertFalse(cfg.adaptive.enabled)
        self.assertFalse(cfg.debug.enabled)
        self.assertFalse(cfg.output.virtual_camera_enabled)

    def test_performance_profile(self):
        """Verify performance profile reduces detection frequency and disables optional parsing."""
        cfg = AppConfig()
        cfg.apply_profile(ProfileType.PERFORMANCE)
        self.assertEqual(cfg.detection.interval, 6)
        self.assertFalse(cfg.parsing.enabled)
        self.assertFalse(cfg.hair.enabled)

    def test_quality_profile(self):
        """Verify quality profile enables higher frequency detection and parsing."""
        cfg = AppConfig()
        cfg.apply_profile(ProfileType.QUALITY)
        self.assertEqual(cfg.detection.interval, 2)
        self.assertTrue(cfg.parsing.enabled)
        self.assertTrue(cfg.hair.enabled)


if __name__ == "__main__":
    unittest.main()
