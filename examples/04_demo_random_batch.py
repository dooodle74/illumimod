import os
import numpy as np
from PIL import Image
import random
import math

from illumimod.masks import light_to_mask, combine_masks
from illumimod.apply import apply_additive
from illumimod.sampling import sample_pixel_index as sample_point

PATH_IN  = "examples/data/speckle.png"
OUT_DIR   = "examples/out/batch"

W = 500
H = 500
LIGHTS_MIN = 1
LIGHTS_MAX = 10
SIZE_MIN = 1
SIZE_MAX = 5
SAMPLE_MEAN = 250
SAMPLE_STD = 250
BRIGHTNESS_MAX = 120
BRIGHTNESS_MIN = 40
HALFLIFE_MAX = 125
HALFLIFE_MIN = 40
POWER_MAX = 25
POWER_MIN = 10
EXPOSURE_MIN = 7
EXPOSURE_MAX = 13

img = np.array(Image.open(PATH_IN).convert("RGB"))

for i in range(100, 200):
    num_lights = random.randint(LIGHTS_MIN, LIGHTS_MAX)
    scale = math.sqrt(num_lights)
    lights = []
    for light in range(num_lights):
        x = sample_point(mean=SAMPLE_MEAN, std=SAMPLE_STD, length=500)
        y = sample_point(mean=SAMPLE_MEAN, std=SAMPLE_STD, length=500)
        lights.append(
            dict(
                center = (x,y),
                peak = random.randint(BRIGHTNESS_MIN, BRIGHTNESS_MAX),
                core_radius = random.randint(SIZE_MIN, SIZE_MAX),
                half_life_radius = random.randint(HALFLIFE_MIN, HALFLIFE_MAX),
                power = random.randint(POWER_MIN, POWER_MAX)/float(10)
            )
        )
    # build masks and combine additively
    masks = [light_to_mask(W, H, dtype=np.float32, **ld) for ld in lights]
    L = combine_masks(masks)  # purely additive sum, no normalization

    PATH_MASK = os.path.join(OUT_DIR, f"mask-{i}.png")
    PATH_OUT  = os.path.join(OUT_DIR, f"out-{i}.png")

    # quick 8-bit visualization of the mask
    Image.fromarray(np.clip(L, 0, 255).astype(np.uint8)).save(PATH_MASK)

    # apply as-is (assumes mask is already scaled how you want)
    out = apply_additive(img, L, clip=True, exposure=random.randint(EXPOSURE_MIN, EXPOSURE_MAX)/float(10))
    Image.fromarray(out).save(PATH_OUT)

    print("Wrote:")
    print("-", PATH_MASK)
    print("-", PATH_OUT)
