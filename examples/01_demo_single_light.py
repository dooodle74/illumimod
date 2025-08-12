import os
import numpy as np
from PIL import Image
from illumimod.masks import generate_light as light_to_mask

# --- hardcoded I/O ---
PATH_IN  = "examples/data/speckle.png"
PATH_OUT = "examples/out/speckle-01-demo.png"
PATH_MASK = "examples/out/speckle-01-mask_u8.png"

# --- light params ---
PEAK        = 160.0      # center brightness to add
CORE_RADIUS = 12.0       # flat core radius (px)
HALF_LIFE   = 80.0       # distance beyond core where intensity halves
POWER       = 2.0        # tail steepness (1 softer, 2 ≈ gaussian-ish, >2 sharper)
SOFT_EDGE   = 0.0        # px of core-edge smoothing
ANGLE_DEG   = 90.0       # 90=perpendicular to surface
DIRECTION   = 0.0        # 0=from up (north), 90=from right (east), clockwise
LAMBERTIAN  = False      # keep False if you want exact PEAK at center

def main():
    os.makedirs(os.path.dirname(PATH_OUT), exist_ok=True)

    # load image (RGB)
    img = np.array(Image.open(PATH_IN).convert("RGB"))
    H, W = img.shape[:2]
    center = (W / 2.0, H / 2.0)  # light at image center

    # generate mask (float32, same HxW)
    mask = light_to_mask(
        width=W, height=H, center=center,
        peak=PEAK, core_radius=CORE_RADIUS, half_life_radius=HALF_LIFE,
        power=POWER, edge_softness=SOFT_EDGE,
        angle_deg=ANGLE_DEG, direction_deg=DIRECTION,
        lambertian=LAMBERTIAN, dtype=np.float32
    )

    # save a quick 8-bit view of the mask
    Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8)).save(PATH_MASK)

    # simplest preview: ADDITIVE application (just to see the effect)
    out = np.clip(img.astype(np.float32) + mask[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(out).save(PATH_OUT)

    print("wrote:")
    print("-", PATH_MASK)
    print("-", PATH_OUT)

if __name__ == "__main__":
    main()
