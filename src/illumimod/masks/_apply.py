import numpy as np
import numpy.typing as npt

def apply(
    img: npt.NDArray,           # (H,W) or (H,W,3)
    mask: npt.NDArray,          # (H,W), same spatial size as img
    *,
    exposure: float = 1.0,
    clip: bool = True,
    out_dtype = np.uint8,
) -> npt.NDArray:
    """
    Add the mask to the image (optionally after an exposure gain), then clip.

    This is an **additive** apply with no automatic scaling. The mask is assumed to be
    in the correct units for the image (e.g., 0–255 for ``uint8`` sRGB). For RGB images,
    the mask is broadcast across channels.

    Parameters
    ----------
    img : ndarray
        Input image, ``(H, W)`` (grayscale) or ``(H, W, 3)`` (RGB).
    mask : ndarray
        Additive mask, shape ``(H, W)``.
    exposure : float, optional
        Global gain applied to the base image before addition (``out = exposure*img + mask``).
    clip : bool, optional
        If ``True``, clamp the result into ``[0, 255]`` for integer inputs, else ``[0, 1]``.
    out_dtype : numpy dtype, optional
        Output dtype (default ``np.uint8``).

    Returns
    -------
    ndarray
        Processed image with the same spatial shape as ``img`` and dtype ``out_dtype``.

    Raises
    ------
    ValueError
        If ``mask`` and ``img`` spatial dimensions differ.

    Notes
    -----
    For more physically plausible results, consider performing apply in **linear light**
    in a future version (convert sRGB ↔ linear around this operation).
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