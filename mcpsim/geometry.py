"""Detector geometry: cylindrical and rectangular bar-array models.

Provides the polar half-angle used by the bremsstrahlung acceptance cut and the
geometry arguments passed to the C++ decay step. The C++ decay code
(lanl_decayPion_12bar.cc) computes the rectangular-face acceptance directly from
distance + bar layout; here we only assemble the parameters and the effective
angle the brem (theta, p) grids are cut at.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import DetectorConfig


def cylindrical_theta_cut(radius_m: float, distance_m: float) -> float:
    """Polar half-angle subtended by a cylindrical detector face."""
    return np.arctan(radius_m / distance_m)


def bar_array_dims(det: DetectorConfig) -> Tuple[float, float]:
    """Front-face (width, height) of a bar array, in metres."""
    width = det.bar_columns * det.bar_size_m
    height = det.bar_rows * det.bar_size_m
    return width, height


def equivalent_theta_cut(det: DetectorConfig) -> float:
    """Equal-area cone half-angle for a rectangular face.

    The precomputed brem grids are azimuthally symmetric in (theta, p), so a
    rectangular aperture is approximated by the disk of equal area:
    r_equiv = sqrt(width * height / pi). Used only when applying brem to a
    bar-array detector (an unusual combination; brem is normally cylindrical).
    """
    width, height = bar_array_dims(det)
    r_equiv = np.sqrt(width * height / np.pi)
    return np.arctan(r_equiv / det.distance_m)


def acceptance_theta_cut(det: DetectorConfig) -> float:
    """Polar half-angle used to cut the bremsstrahlung (theta, p) grids."""
    if det.type == "cylindrical":
        return cylindrical_theta_cut(det.radius_m, det.distance_m)
    if det.type == "bar_array":
        return equivalent_theta_cut(det)
    raise ValueError(f"unknown detector type {det.type!r} "
                     f"(valid: 'cylindrical', 'bar_array')")


def offaxis_offset_m(det: DetectorConfig) -> float:
    """Transverse displacement of the face centre on the detector plane."""
    return det.distance_m * np.tan(1e-3 * det.offaxis_mrad)


def circle_hit_fraction(r: np.ndarray, d: float, radius: float) -> np.ndarray:
    """Azimuth-averaged probability of landing on a circular face of radius
    `radius` centred at transverse offset d (cylindrical detector analogue of
    face_hit_fraction). On axis (d = 0) this reduces to the sharp r <= radius
    cut; off axis it is the circle-circle arc fraction."""
    r = np.asarray(r, dtype=float)
    if d == 0:
        return (r <= radius).astype(float)
    f = np.zeros_like(r)
    at_origin = r == 0
    if np.any(at_origin):
        f[at_origin] = 1.0 if d <= radius else 0.0
    inside = (r > 0) & (r <= radius - d)          # ring fully inside the face
    f[inside] = 1.0
    m = (r > max(d - radius, 0)) & (r < d + radius) & ~inside & (r > 0)
    if np.any(m):
        rm = r[m]
        cosphi = (rm * rm + d * d - radius * radius) / (2.0 * rm * d)
        f[m] = np.arccos(np.clip(cosphi, -1.0, 1.0)) / np.pi
    return f


def face_hit_fraction(r: np.ndarray, d: float, half_w: float,
                      half_h: float) -> np.ndarray:
    """Azimuth-averaged probability that a particle crossing the detector
    plane at radius r lands on a rectangular face.

    The face spans x in [d - half_w, d + half_w], |y| <= half_h, with d >= 0
    the transverse off-axis offset of its centre. For azimuthally symmetric
    production a particle at radius r is uniform in phi, so the hit
    probability is the fraction of the circle of radius r inside the face —
    an exact, smooth kernel that replaces binary hit-counting on a tiny face
    (variance reduction ~ 2*pi*d / face width off-axis).

    Exact for every offset including d = 0 and r < half-width: the arc is the
    intersection of the x-band [phi1, phi2] (phi1 = arccos((d+w)/r),
    phi2 = arccos((d-w)/r), clipped) with the y-band [0, phi_y] and its
    mirror [pi - phi_y, pi], doubled for +-phi. (The FLAME script's original
    version dropped the mirror branch, halving the on-axis answer.)
    """
    r = np.asarray(r, dtype=float)
    f = np.zeros_like(r)

    # r == 0: the origin is on the face iff it contains (0, 0).
    at_origin = r == 0
    if np.any(at_origin):
        f[at_origin] = 1.0 if (abs(d) <= half_w) else 0.0

    m = (r > 0) & (r > d - half_w) & (r < np.hypot(d + half_w, half_h))
    if not np.any(m):
        return f
    rm = r[m]
    phi1 = np.arccos(np.clip((d + half_w) / rm, -1.0, 1.0))
    phi2 = np.arccos(np.clip((d - half_w) / rm, -1.0, 1.0))
    phi_y = np.arcsin(np.clip(half_h / rm, 0.0, 1.0))
    main = np.clip(np.minimum(phi2, phi_y) - phi1, 0.0, None)
    mirror = np.clip(phi2 - np.maximum(phi1, np.pi - phi_y), 0.0, None)
    f[m] = (main + mirror) / np.pi
    return f


