import os
import numpy as np
from PIL import Image
from illumimod.masks import light_to_mask, combine_masks

# --- hardcoded I/O ---
PATH_IN  = "examples/data/speckle.png"
PATH_OUT = "examples/out/speckle-02-demo.png"
PATH_MASK = "examples/out/speckle-02-mask_u8.png"

# --- light 1 ---
PEAK_1        = 160.0      # center brightness to add
CORE_RADIUS_1 = 12.0       # flat core radius (px)
HALF_LIFE_1   = 80.0       # distance beyond core where intensity halves
POWER_1       = 2.0        # tail steepness (1 softer, 2 ≈ gaussian-ish, >2 sharper)
SOFT_EDGE_1   = 0.0        # px of core-edge smoothing
ANGLE_DEG_1   = 90.0       # 90=perpendicular to surface
DIRECTION_1   = 0.0        # 0=from up (north), 90=from right (east), clockwise
LAMBERTIAN_1  = False      # keep False if you want exact PEAK at center

# --- light 2 ---
PEAK_2        = 100.0      # center brightness to add
CORE_RADIUS_2 = 1.0       # flat core radius (px)
HALF_LIFE_2   = 20.0       # distance beyond core where intensity halves
POWER_2       = 1.0        # tail steepness (1 softer, 2 ≈ gaussian-ish, >2 sharper)
SOFT_EDGE_2   = 0.0        # px of core-edge smoothing
ANGLE_DEG_2   = 10.0       # 90=perpendicular to surface
DIRECTION_2   = 45       # 0=from up (north), 90=from right (east), clockwise
LAMBERTIAN_2  = False      # keep False if you want exact PEAK at center


'''
ISSUE: LIGHT DIRECTION DOESNT SEEM TO HAVE NEGATIVE AFFECT WHERE IT CAME
'''

def main():
    os.makedirs(os.path.dirname(PATH_OUT), exist_ok=True)

    # load image (RGB)
    img = np.array(Image.open(PATH_IN).convert("RGB"))
    H, W = img.shape[:2]
    center_1 = (W / 2.0, H / 2.0)  
    center_2 = (W / 2.0 + W / 4.0, H / 2.0 + H / 4.0)

    # generate mask 1
    mask_1 = light_to_mask(
        width=W, height=H, center=center_1,
        peak=PEAK_1, core_radius=CORE_RADIUS_1, half_life_radius=HALF_LIFE_1,
        power=POWER_1, edge_softness=SOFT_EDGE_1,
        angle_deg=ANGLE_DEG_1, direction_deg=DIRECTION_1,
        lambertian=LAMBERTIAN_1, dtype=np.float32
    )

    # generate mask 2
    mask_2 = light_to_mask(
        width=W, height=H, center=center_1,
        peak=PEAK_2, core_radius=CORE_RADIUS_2, half_life_radius=HALF_LIFE_2,
        power=POWER_2, edge_softness=SOFT_EDGE_2,
        angle_deg=ANGLE_DEG_2, direction_deg=DIRECTION_2,
        lambertian=LAMBERTIAN_2, dtype=np.float32
    )

    # combine masks
    mask = combine_masks([ mask_2])

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
