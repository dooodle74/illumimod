from typing import Any, Dict, Tuple, Optional
import numpy as np
import numpy.typing as npt

# --- brightness channel ---

def get_brightness_channel(
    arr: npt.NDArray,
    luma: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
) -> npt.NDArray:
    """
    Return (H,W) brightness from gray/RGB/any-C image.
    (H,W)        -> passthrough
    (H,W,1)      -> squeeze
    (H,W,3) RGB  -> Rec.709 luma
    (H,W,C!=3,1) -> channel-wise mean
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[..., 0]
        if arr.shape[2] == 3:
            r, g, b = luma
            x = arr.astype(np.float32, copy=False)
            y = r * x[..., 0] + g * x[..., 1] + b * x[..., 2]
            return y.astype(arr.dtype, copy=False) if np.issubdtype(arr.dtype, np.integer) else y
        return arr.mean(axis=2)
    return arr

def get_brightness_stats(
    img: npt.NDArray,
    *,
    luma: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
) -> Dict[str, float]:
    """
    Return a dict with HARD-CODED keys:
      mean, std, p0, p1, p25, p50, p75, p99, p100
    """
    # brightness vector (float32)
    bright = get_brightness_channel(img, luma=luma).astype(np.float32, copy=False).ravel()
    if bright.size == 0:
        return {"mean": np.nan, "std": np.nan, "p0": np.nan, "p1": np.nan, "p25": np.nan,
                "p50": np.nan, "p75": np.nan, "p99": np.nan, "p100": np.nan}

    vmin = float(np.min(bright))
    vmax = float(np.max(bright))
    p1, p25, p50, p75, p99 = np.percentile(bright, [1, 25, 50, 75, 99]).astype(np.float32)

    stats = {
        "mean": float(np.mean(bright)),
        "std": float(np.std(bright)),
        "p0": vmin,
        "p1": float(p1),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p99": float(p99),
        "p100": vmax,
    }
    return stats

