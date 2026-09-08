"""Read-only fixtures for a skill behavior exercise, not production library code."""
import math


def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def quantize_row(row):
    if not row or not all(math.isfinite(v) for v in row):
        raise ValueError('a nonempty finite row is required')
    low, high = min(row), max(row)
    scale = (high - low) / 255 if high != low else 1.0
    codes = [min(255, max(0, round((v - low) / scale))) for v in row]
    return codes, scale, low


def reconstruct(codes, scale, low):
    return [low + scale * code for code in codes]


PARSERS = {'int': int, 'float': float}


def parse_value(kind, text):
    parser = PARSERS.get(kind)
    if parser is None:
        raise ValueError('unsupported kind')
    return parser(text)
