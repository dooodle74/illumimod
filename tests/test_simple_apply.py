import numpy as np
from illumimod.apply import apply_additive

def test_apply_additive_center_brightens():
    H = W = 64
    img = np.full((H, W), 100, np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[32, 32] = 80.0

    out, stats = apply_additive(img, mask)
    assert out[32, 32] == 180
    assert out[0, 0] == 100
    assert stats["mode"] == "fixed"

def test_apply_additive_rgb_broadcast_and_clip():
    img = np.full((4, 4, 3), 250, np.uint8)
    mask = np.full((4, 4), 20, np.float32)  # +20 everywhere
    out, _ = apply_additive(img, mask, clip=True)
    # 250 + 20 → clipped to 255
    assert out.max() == 255
    assert out.dtype == np.uint8