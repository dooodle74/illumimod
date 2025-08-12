# examples/demo_image.py
import os
import numpy as np
from PIL import Image

# If your function is named radial_light_mask, change the import accordingly:
from illumimod.masks import light_to_mask as light_mask, combine_masks
from illumimod.apply import apply_additive

PATH_IN  = "examples/data/speckle.png"
OUT_DIR   = "examples/out"
PATH_MASK = os.path.join(OUT_DIR, "speckle-03-mask.png")
PATH_OUT  = os.path.join(OUT_DIR, "speckle-03-result_additive.png")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load image as RGB
    img = np.array(Image.open(PATH_IN).convert("RGB"))
    H, W = img.shape[:2]

    # two arbitrary lights (tweak to taste)
    lights = [
        # center spot, perpendicular
        dict(center=(0.35 * W, 0.60 * H), peak=140.0, core_radius=12.0, half_life_radius=80.0,
             power=2.0, edge_softness=0.0, angle_deg=90.0, direction_deg=0.0, lambertian=False),
        # raking light from the right
        dict(center=(0.70 * W, 0.30 * H), peak= 200.0, core_radius= 8.0, half_life_radius=60.0,
             power=1.5, edge_softness=0.0, angle_deg=60.0, direction_deg=90.0, lambertian=False),
    ]

    # build masks and combine additively
    masks = [light_mask(W, H, dtype=np.float32, **ld) for ld in lights]
    L = combine_masks(masks)  # purely additive sum, no normalization

    # quick 8-bit visualization of the mask
    Image.fromarray(np.clip(L, 0, 255).astype(np.uint8)).save(PATH_MASK)

    # apply as-is (assumes mask is already scaled how you want)
    out = apply_additive(img, L, clip=True)
    Image.fromarray(out).save(PATH_OUT)

    print("Wrote:")
    print("-", PATH_MASK)
    print("-", PATH_OUT)

if __name__ == "__main__":
    main()
