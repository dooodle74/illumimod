import numpy as np
import numpy.typing as npt

def apply_additive(
    img: npt.NDArray,           # (H,W) or (H,W,3)
    mask: npt.NDArray,          # (H,W), same spatial size as img
    *,
    exposure: float = 1.0,
    clip: bool = True,
    out_dtype = np.uint8,
) -> npt.NDArray:
    """
    Add the mask to the image as-is (no auto-scaling), then clip.

    Assumptions:
      - mask is already in the correct units (e.g., 0..255 if img is uint8 sRGB).
      - mask is non-negative (lights add). Negative values will darken.

    Returns image after mask.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("apply_additive: mask HxW must match image HxW")

    img_f  = img.astype(np.float32, copy=False)
    add    = mask.astype(np.float32, copy=False)

    # Broadcast to RGB if needed
    if img_f.ndim == 3 and img_f.shape[2] != 1 and add.ndim == 2:
        add = add[..., None]

    out = exposure * img_f + add

    if clip:
        headroom = 255.0 if np.issubdtype(img.dtype, np.integer) else 1.0
        out = np.clip(out, 0.0, headroom)

    return out.astype(out_dtype, copy=False)
