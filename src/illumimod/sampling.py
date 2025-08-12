import numpy as np
RNG = np.random.default_rng(seed=74)

def sample_truncated_normal(mean, std, low=0.0, high=1.0, max_tries=1000):
    """
    Sample a single value from a normal distribution with the given mean and std dev,
    truncated to [low, high].

    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the distribution (controls spread).
    low : float
        Minimum allowed value (inclusive).
    high : float
        Maximum allowed value (inclusive).
    max_tries : int
        Max number of rejection attempts before falling back to clipping.

    Returns
    -------
    float
        Sampled value within [low, high].
    """
    for _ in range(max_tries):
        x = RNG.normal(loc=mean, scale=std)
        if low <= x <= high:
            return x

    # Fallback: clip if we exceeded max_tries
    return float(np.clip(x, low, high))


def sample_pixel_index(length, mean, std, normalized=False, clip=True, resample_tries=1000):
    """
    Sample a pixel index along a 1D axis.

    Parameters
    ----------
    length : int
        Number of pixels along the axis.
    mean : float
        Mean of the distribution (pixels if normalized=False, normalized if True).
    std : float
        Standard deviation (pixels if normalized=False, normalized if True).
    normalized : bool
        If True, mean/std are in normalized units [0,1].
        If False, mean/std are in pixel units.
    clip : bool
        If True, perform one draw and clip (max_tries=1).
        If False, do truncated sampling (max_tries=tries).
    resample_tries : int
        Maximum attempts for truncated sampling (ignored if clip=True).

    Returns
    -------
    int
        Pixel index in range [0, length-1].
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

    # Choose number of tries
    tries = 1 if clip else resample_tries

    # Call gaussian generator
    value = sample_truncated_normal(mean_px, std_px, low=0.0, high=float(length - 1), max_tries=tries)

    return int(round(value))
