# --- optional helpers for lookups (normalize keys) ---

def normalize_stat_key(key: Any) -> str:
    """
    Map various aliases to the fixed keys above.
      - numbers -> percentiles (25 -> 'p25', 99 -> 'p99')
      - '25%', 'p25', 'P25', 'p01' -> 'p25' / 'p1'
      - 'min'/'maximum' -> 'p0'/'p100'
      - 'median'/'med'  -> 'p50'
      - 'avg'/'average' -> 'mean'
      - 'stdev'/'sigma' -> 'std'
    """
    if isinstance(key, (int, float)):
        q = int(round(float(key)))
        return f"p{q}"

    s = str(key).strip().lower()
    if s.endswith("%"):
        s = s[:-1].strip()
    if s.startswith("p"):
        s = s[1:]
    if s in ("min", "minimum"):  return "p0"
    if s in ("max", "maximum"):  return "p100"
    if s in ("median", "med"):   return "p50"
    if s in ("avg", "average"):  return "mean"
    if s in ("stdev", "stddev", "σ", "sigma"): return "std"

    # percentiles like "01" or "1" -> p1
    try:
        q = int(round(float(s)))
        return f"p{q}"
    except ValueError:
        pass

    # fall through: allow exact fixed keys if already correct
    if s in {"mean","std","p0","p1","p25","p50","p75","p99","p100"}:
        return s
    raise KeyError("Unknown stat key: %r" % key)

def get_stat(stats: Dict[str, float], key: Any, default: Optional[float] = None) -> Optional[float]:
    """Convenience accessor that normalizes the key first."""
    try:
        k = normalize_stat_key(key)
    except KeyError:
        return default
    return stats.get(k, default)
