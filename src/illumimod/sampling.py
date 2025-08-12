import numpy as np

RNG: np.random.Generator = np.random.default_rng(seed=74)

def sample_truncated_normal(
    mean: float,
    std: float,
    low: float = 0.0,
    high: float = 1.0,
    max_tries: int = 1000,
) -> float:
    """
    Sample a single value from a normal distribution truncated to ``[low, high]``.

    Parameters
    ----------
    mean : float
        Mean of the (untruncated) normal distribution.
    std : float
        Standard deviation of the (untruncated) normal distribution.
    low : float, optional
        Inclusive lower bound of the truncation interval. Default is 0.0.
    high : float, optional
        Inclusive upper bound of the truncation interval. Default is 1.0.
    max_tries : int, optional
        Maximum number of rejection-sampling attempts. If the budget is
        exhausted, the last draw is clipped into ``[low, high]``. Default is 1000.

    Returns
    -------
    float
        A sampled value within ``[low, high]``.

    Notes
    -----
    - If ``max_tries == 1``, this behaves like "draw once and clip".
    - For larger ``max_tries``, it uses simple rejection sampling; only if all
      attempts fail will the result be clipped.
    """
    for _ in range(max_tries):
        x = RNG.normal(loc=mean, scale=std)
        if low <= x <= high:
            return float(x)

    # Fallback: clip if we exceeded max_tries - x unbound error fine
    return float(np.clip(x, low, high))

def sample_pixel_index(
    length: int,
    mean: float,
    std: float,
    *,
    normalized: bool = False,
    clip: bool = True,
    resample_tries: int = 1000,
) -> int:
    """
    Sample a pixel index along a 1D axis using a (possibly truncated) normal draw.

    Parameters
    ----------
    length : int
        Number of pixels along the axis (must be > 0). Valid output indices are
        in ``[0, length-1]``.
    mean : float
        Mean of the distribution. Interpreted as pixels if ``normalized=False``;
        as normalized units in ``[0, 1]`` if ``normalized=True``.
    std : float
        Standard deviation. Interpreted in the same units as ``mean``.
    normalized : bool, optional
        If ``True``, treat ``mean``/``std`` as normalized to the axis length
        (i.e., multiply by ``length-1`` internally). Default is ``False``.
    clip : bool, optional
        If ``True``, perform a single draw and allow clipping into bounds
        (equivalent to calling ``sample_truncated_normal(..., max_tries=1)``).
        If ``False``, perform rejection sampling with up to ``resample_tries``
        attempts. Default is ``True``.
    resample_tries : int, optional
        Maximum attempts for rejection sampling when ``clip=False``. Ignored if
        ``clip=True``. Default is 1000.

    Returns
    -------
    int
        A pixel index in ``[0, length-1]`` (rounded to nearest integer).

    Raises
    ------
    ValueError
        If ``length <= 0``.

    Notes
    -----
    - When ``normalized=True``, the internal mean/std are scaled by ``(length-1)``.
    - The sampled (float) value is rounded to the nearest integer before
      returning.
    - With ``clip=True``, a single out-of-range draw is simply clamped into the
      valid interval; with ``clip=False``, the function tries to draw a valid
      value up to ``resample_tries`` times before falling back to clamping.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    # Convert normalized to pixel units
    if normalized:
        mean_px = mean * (length - 1)
        std_px = std * (length - 1)
    else:
        mean_px = mean
        std_px = std

    # Number of tries for truncated sampling
    tries = 1 if clip else resample_tries

    # Draw within [0, length-1]
    value = sample_truncated_normal(
        mean=mean_px,
        std=std_px,
        low=0.0,
        high=float(length - 1),
        max_tries=tries,
    )

    return int(round(value))