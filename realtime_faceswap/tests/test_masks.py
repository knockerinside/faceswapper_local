"""Unit tests for Face Masking and Temporal Smoothing (Section 13 & 17)."""
import unittest
import numpy as np
from realtime_faceswap.compositing.masks import MaskGenerator
from realtime_faceswap.hair.hair_effects import HairProcessor


class TestMasks(unittest.TestCase):
    def test_mask_generation_dimensions_and_range(self):
        """Verify 128x128 mask range is bounded strictly in [0.0, 1.0]."""
        gen = MaskGenerator(crop_size=128)
        mask = gen.generate_mask(feather_px=10, erosion_px=2)
        self.assertEqual(mask.shape, (128, 128))
        self.assertEqual(mask.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(mask)), 0.0)
        self.assertLessEqual(float(np.max(mask)), 1.0)
        self.assertGreater(float(np.sum(mask)), 100.0)

    def test_hair_boundary_protection(self):
        """Verify that hair region is subtracted from face mask to avoid overwriting hair."""
        hair_proc = HairProcessor()
        face_mask = np.full((128, 128), 255, dtype=np.uint8)
        # Hair mask in top region
        hair_mask = np.zeros((128, 128), dtype=np.uint8)
        hair_mask[:30, :] = 255

        protected = hair_proc.protect_hair_boundary(face_mask, hair_mask, dilation_px=0)
        # Top region should now be 0 (cleared out)
        self.assertEqual(int(protected[10, 64]), 0)
        # Lower face remains 255
        self.assertEqual(int(protected[80, 64]), 255)


if __name__ == "__main__":
    unittest.main()
