"""
ringed_transit.py
-----------------
Model transit light curves of a ringed exoplanet.

Physics / geometry
  * Keplerian orbit (eccentric allowed): P, t0, a/R*, i, e, omega.
  * Star: quadratic limb darkening, I(mu) = 1 - u1(1-mu) - u2(1-mu)^2.
  * Planet: opaque disk of radius Rp/R*.
  * Ring: flat, thin annulus [r_in, r_out] (in planet radii) with normal
    optical depth tau. Seen at angle theta to the line of sight, its
    transmission is exp(-tau / |cos theta|).  Where ring and planet overlap
    on the sky, the planet (opaque) wins.
  * Ring orientation is fixed in space during transit, so its sky
    projection is constant. You can give it either
      - in the sky frame: ring_incl (0 = face-on, 90 = edge-on) and
        ring_pa (position angle of the normal's sky projection, measured
        from the direction of orbital motion toward sky-north), or
      - in the orbit frame: obliquity (tilt of ring normal from orbital
        angular momentum) and ring_azimuth (longitude of that tilt, 0 =
        tilted toward the direction of motion at mid-transit).

Numerics
  The occulter (planet + ring) is rasterised once into a transmission map
  on a grid centred on the planet. At each time the map is placed over the
  limb-darkened stellar disk and the blocked flux is summed. Increase
  `n_grid` for accuracy (error ~ 1/n_grid^2 in depth for smooth edges).

Coordinates (Winn 2010 convention), units of stellar radius:
  x : along orbital motion at mid-transit, y : sky "north", z : toward observer.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Orbit
# ----------------------------------------------------------------------------
def _solve_kepler(M, e, tol=1e-12, maxiter=50):
    """Solve E - e sin E = M (Newton), vectorised."""
    M = np.mod(M, 2 * np.pi)
    E = M + e * np.sin(M) if e < 0.8 else np.full_like(M, np.pi)
    for _ in range(maxiter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def sky_position(t, P, t0, a, inc_deg, e=0.0, omega_deg=90.0):
    """
    Planet position relative to the star centre (units of R*).
    t0 is the time of mid-transit (conjunction).
    Returns X, Y, Z ; Z > 0 means the planet is in front of the star.
    """
    t = np.asarray(t, dtype=float)
    inc = np.radians(inc_deg)
    w = np.radians(omega_deg)

    # true -> mean anomaly at conjunction so that t0 is mid-transit
    f_c = np.pi / 2 - w
    E_c = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f_c / 2))
    M_c = E_c - e * np.sin(E_c)

    M = 2 * np.pi * (t - t0) / P + M_c
    E = _solve_kepler(M, e)
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                       np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))

    u = w + f
    X = -r * np.cos(u)
    Y = -r * np.sin(u) * np.cos(inc)
    Z = r * np.sin(u) * np.sin(inc)
    return X, Y, Z


# ----------------------------------------------------------------------------
# Ring orientation helpers
# ----------------------------------------------------------------------------
def ring_normal_from_sky(ring_incl_deg, ring_pa_deg):
    """Unit ring normal from sky-frame angles."""
    th = np.radians(ring_incl_deg)
    pa = np.radians(ring_pa_deg)
    return np.array([np.sin(th) * np.cos(pa),
                     np.sin(th) * np.sin(pa),
                     np.cos(th)])


def ring_normal_from_orbit(obliquity_deg, ring_azimuth_deg, inc_deg):
    """
    Unit ring normal from orbit-frame angles.
    obliquity: angle between ring normal and orbital angular momentum.
    ring_azimuth: azimuth of the tilt around the orbit normal
                  (0 = toward the direction of motion at mid-transit,
                   90 = toward the observer-side of the orbit plane).
    """
    i = np.radians(inc_deg)
    ob = np.radians(obliquity_deg)
    az = np.radians(ring_azimuth_deg)
    e1 = np.array([1.0, 0.0, 0.0])                 # motion at transit
    e3 = np.array([0.0, np.sin(i), np.cos(i)])     # orbital ang. momentum
    e2 = np.cross(e3, e1)
    n = (np.sin(ob) * np.cos(az) * e1 + np.sin(ob) * np.sin(az) * e2
         + np.cos(ob) * e3)
    return n / np.linalg.norm(n)


def sky_angles_from_normal(n):
    """Return (ring_incl_deg, ring_pa_deg) for a unit normal."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    if n[2] < 0:                       # n and -n describe the same ring
        n = -n
    return (np.degrees(np.arccos(np.clip(n[2], -1, 1))),
            np.degrees(np.arctan2(n[1], n[0])) % 360)


# ----------------------------------------------------------------------------
# Occulter transmission map
# ----------------------------------------------------------------------------
def occulter_map(rp, r_in, r_out, tau, normal, n_grid=400, oblateness=0.0):
    """
    Rasterise planet + ring on a square grid centred on the planet.

    rp      : planet radius / R*
    r_in    : inner ring radius / Rp   (set r_out <= r_in or tau=0 for no ring)
    r_out   : outer ring radius / Rp
    tau     : normal optical depth of the ring (np.inf = opaque)
    normal  : unit ring-normal vector in the sky frame (also the planet's
              spin axis)
    oblateness : planet flattening f = 1 - R_pol/R_eq (Saturn 0.098); the
              disk becomes an ellipse with axis ratio
              sqrt(cos^2 th + (1-f)^2 sin^2 th), th = spin-axis angle to the
              line of sight; rp is the equatorial radius.

    Returns gx, gy (1-D offsets of non-transparent pixels, units R*),
            opacity (1 - transmission) for those pixels, and pixel area.
    """
    has_ring = (r_out > r_in) and (tau > 0)
    half = rp * (max(r_out, 1.0) if has_ring else 1.0)
    edges = np.linspace(-half, half, n_grid + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    dA = (edges[1] - edges[0]) ** 2
    X, Y = np.meshgrid(c, c)

    opacity = np.zeros_like(X)

    if has_ring:
        nx, ny, nz = normal
        nz_eff = nz if abs(nz) > 1e-9 else 1e-9 * (1 if nz >= 0 else -1)
        # intersection of the line of sight with the ring plane
        Zr = -(nx * X + ny * Y) / nz_eff
        d = np.sqrt(X**2 + Y**2 + Zr**2) / rp
        in_ring = (d >= r_in) & (d <= r_out)
        trans = np.exp(-tau / abs(nz_eff)) if np.isfinite(tau) else 0.0
        opacity[in_ring] = 1.0 - trans

    if oblateness > 0:
        nx, ny, nz = normal
        q = np.sqrt(nz**2 + (1 - oblateness) ** 2 * (1 - nz**2))
        ps = np.hypot(nx, ny)
        ux, uy = (nx / ps, ny / ps) if ps > 1e-9 else (0.0, 1.0)
        u = X * ux + Y * uy                  # along projected spin axis
        v = -X * uy + Y * ux
        opacity[(v / rp) ** 2 + (u / (rp * q)) ** 2 <= 1.0] = 1.0
    else:
        opacity[X**2 + Y**2 <= rp**2] = 1.0    # planet is opaque

    keep = opacity > 0
    return X[keep], Y[keep], opacity[keep], dA


# ----------------------------------------------------------------------------
# Light curve
# ----------------------------------------------------------------------------
def light_curve(t, P, t0, a, inc, rp, u1=0.4, u2=0.25, e=0.0, omega=90.0,
                r_in=1.2, r_out=2.0, tau=1.0,
                ring_incl=None, ring_pa=None,
                obliquity=None, ring_azimuth=None, oblateness=0.0,
                n_grid=400, exptime=None, supersample=1, chunk=200):
    """
    Normalised flux of a star transited by a ringed planet.

    t        : times (same units as P, t0)
    P, t0    : period, mid-transit time
    a        : semi-major axis / R*
    inc      : orbital inclination [deg]
    rp       : planet radius / R*
    u1, u2   : quadratic limb-darkening coefficients
    e, omega : eccentricity, argument of periastron [deg]
    r_in, r_out : ring radii in planet radii
    tau      : ring normal optical depth (0 = no ring, np.inf = opaque)
    Ring orientation (give ONE pair):
        ring_incl, ring_pa          (sky frame, deg)
        obliquity, ring_azimuth     (orbit frame, deg)
      default if neither: obliquity=0 (ring in orbital plane).
    oblateness : planet flattening (spin axis = ring normal)
    n_grid   : grid resolution for the occulter
    exptime, supersample : integrate over exposure time (e.g. Kepler LC)
    """
    t = np.atleast_1d(np.asarray(t, float))

    # ring orientation
    if ring_incl is not None:
        n = ring_normal_from_sky(ring_incl, ring_pa or 0.0)
    else:
        n = ring_normal_from_orbit(obliquity or 0.0, ring_azimuth or 0.0, inc)

    gx, gy, opac, dA = occulter_map(rp, r_in, r_out, tau, n, n_grid,
                                    oblateness)
    reach = np.sqrt(gx**2 + gy**2).max() if gx.size else rp

    # exposure-time supersampling
    if exptime and supersample > 1:
        off = (np.arange(supersample) + 0.5) / supersample - 0.5
        tt = (t[:, None] + off[None, :] * exptime).ravel()
    else:
        tt = t
        supersample = 1

    X, Y, Z = sky_position(tt, P, t0, a, inc, e, omega)
    flux = np.ones_like(tt)

    norm = np.pi * (1 - u1 / 3 - u2 / 6)     # total stellar flux
    active = np.where((Z > 0) & (np.hypot(X, Y) < 1 + reach))[0]

    for s in range(0, active.size, chunk):
        idx = active[s:s + chunk]
        xs = X[idx, None] + gx[None, :]
        ys = Y[idx, None] + gy[None, :]
        r2 = xs**2 + ys**2
        on = r2 < 1.0
        mu = np.sqrt(np.where(on, 1.0 - r2, 0.0))
        I = (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) * on
        flux[idx] = 1.0 - (I * opac[None, :]).sum(axis=1) * dA / norm

    return flux.reshape(-1, supersample).mean(axis=1)


# ----------------------------------------------------------------------------
# Eclipsing binaries (astrophysical false positives)
# ----------------------------------------------------------------------------
def _circle_overlap(d, r1, r2):
    """Overlap area of two circles with radii r1, r2 at separation d."""
    d = np.asarray(d, float)
    out = np.zeros_like(d)
    full = d <= abs(r1 - r2)
    out[full] = np.pi * min(r1, r2) ** 2
    part = (d > abs(r1 - r2)) & (d < r1 + r2)
    dp = d[part]
    c1 = np.clip((dp**2 + r1**2 - r2**2) / (2 * dp * r1), -1, 1)
    c2 = np.clip((dp**2 + r2**2 - r1**2) / (2 * dp * r2), -1, 1)
    out[part] = (r1**2 * np.arccos(c1) + r2**2 * np.arccos(c2)
                 - 0.5 * np.sqrt(np.clip((-dp + r1 + r2) * (dp + r1 - r2)
                                         * (dp - r1 + r2) * (dp + r1 + r2),
                                         0, None)))
    return out


def eb_light_curve(t, P, t0, a, inc, k, sb_ratio, u1=0.4, u2=0.25, e=0.0,
                   omega=90.0, ellip_ppm=0.0, refl_ppm=0.0, dilution=0.0,
                   n_grid=150, exptime=None, supersample=1):
    """
    Normalised flux of a detached / grazing eclipsing binary.

    Primary star radius = 1 (all lengths in R1). Companion radius k, surface
    brightness ratio sb_ratio (so L2/L1 ~ sb_ratio k^2).  Primary eclipse at
    t0 (companion in front, limb-darkened primary, reuses light_curve);
    secondary eclipse where the primary covers the (uniform-disk) companion.
    ellip_ppm : ellipsoidal semi-amplitude (minima at both conjunctions)
    refl_ppm  : reflection / heating semi-amplitude (max at secondary)
    dilution  : extra unrelated flux in units of the binary's flux; a
                background / blended EB has dilution >> 1.
    """
    t = np.atleast_1d(np.asarray(t, float))
    prim = light_curve(t, P, t0, a, inc, k, u1=u1, u2=u2, e=e, omega=omega,
                       tau=0.0, n_grid=n_grid, exptime=exptime,
                       supersample=supersample)
    if exptime and supersample > 1:
        off = (np.arange(supersample) + 0.5) / supersample - 0.5
        tt = (t[:, None] + off[None, :] * exptime).ravel()
    else:
        tt, supersample = t, 1
    X, Y, Z = sky_position(tt, P, t0, a, inc, e, omega)
    blk2 = np.where(Z < 0, _circle_overlap(np.hypot(X, Y), 1.0, k)
                    / (np.pi * k**2), 0.0)
    blk2 = blk2.reshape(-1, supersample).mean(axis=1)
    L2 = sb_ratio * k**2
    ph = 2 * np.pi * (t - t0) / P
    var = 1 - ellip_ppm * 1e-6 * np.cos(2 * ph) - refl_ppm * 1e-6 * np.cos(ph)
    f = (prim + L2 * (1 - blk2)) / (1 + L2) * var
    return (f + dilution) / (1 + dilution)


# ----------------------------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------------------------
def plot_sky_view(ax, rp, r_in, r_out, normal, b, n_grid=300):
    """Draw the star and the occulter at mid-transit (impact parameter b)."""
    import matplotlib.pyplot as plt  # noqa: F401
    gx, gy, opac, _ = occulter_map(rp, r_in, r_out, np.inf, normal, n_grid)
    ax.add_patch(__import__("matplotlib").patches.Circle(
        (0, 0), 1, color="gold", alpha=0.6))
    ax.scatter(gx, gy + b, s=0.2, c="k", alpha=0.5, lw=0)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("x [R*]")
    ax.set_ylabel("y [R*]")


# ----------------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------------
def demo_ring_orientations():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A Saturn-like planet around a Sun-like star on a ~1-yr orbit
    pars = dict(P=365.25, t0=0.0, a=215.0, inc=89.85, rp=0.084,
                u1=0.4, u2=0.25, e=0.0, omega=90.0)
    ring = dict(r_in=1.24, r_out=2.27, tau=1.0)

    t = np.linspace(-0.6, 0.6, 1500)   # days

    f_noring = light_curve(t, **pars, tau=0.0)
    cases = {
        "ring incl 70°, PA 30°": dict(ring_incl=70, ring_pa=30),
        "ring incl 85°, PA 0°":  dict(ring_incl=85, ring_pa=0),
        "obliquity 27°, az 45°": dict(obliquity=27, ring_azimuth=45),
    }

    fig = plt.figure(figsize=(12, 7))
    ax_lc = fig.add_subplot(2, 1, 1)
    ax_lc.plot(t * 24, f_noring, "k--", lw=1, label="no ring")
    curves = {}
    for lab, kw in cases.items():
        curves[lab] = light_curve(t, **pars, **ring, **kw)
        ax_lc.plot(t * 24, curves[lab], lw=1.5, label=lab)
    ax_lc.set_xlabel("time from mid-transit [h]")
    ax_lc.set_ylabel("relative flux")
    ax_lc.legend(fontsize=8)
    ax_lc.set_title("Ringed-planet transit")

    b = pars["a"] * np.cos(np.radians(pars["inc"]))
    for k, (lab, kw) in enumerate(cases.items()):
        ax = fig.add_subplot(2, 3, 4 + k)
        if "ring_incl" in kw:
            n = ring_normal_from_sky(kw["ring_incl"], kw["ring_pa"])
        else:
            n = ring_normal_from_orbit(kw["obliquity"], kw["ring_azimuth"],
                                       pars["inc"])
        plot_sky_view(ax, pars["rp"], ring["r_in"], ring["r_out"], n, -b)
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-b - 0.35, -b + 0.35)
        ax.set_title(lab, fontsize=9)

    fig.tight_layout()
    fig.savefig("ringed_transit_example.png", dpi=130)
    print("saved ringed_transit_example.png")


# ============================================================================
#  MOCK OBSERVATIONS: time sampling, stellar activity, noise, systematics
# ============================================================================
#
#  flux = transit(t) * stellar(t) * systematics(t) + red_noise(t) + white_noise
#  stellar(t) = 1 + rotation/spots + granulation + flares
#
#  Everything defaults to ZERO noise / a quiet star, so you can switch on one
#  component at a time.  All times are in days, all *_ppm values are in parts
#  per million of the stellar flux.
# ----------------------------------------------------------------------------

# Observing set-ups.  Any key can be overridden in simulate_lightcurve(**kw).
INSTRUMENTS = {
    # TESS: 27.4-d sectors split into two orbits by a ~1-d downlink gap.
    "tess":     dict(kind="space", cadence_s=120, n_sectors=1,
                     sector_days=27.4, gap_days=1.0,
                     momentum_dump_days=3.1, momentum_dump_len_h=0.25,
                     supersample=1),
    "tess_20s": dict(kind="space", cadence_s=20, n_sectors=1,
                     sector_days=27.4, gap_days=1.0,
                     momentum_dump_days=3.1, momentum_dump_len_h=0.25,
                     supersample=1),
    "tess_ffi": dict(kind="space", cadence_s=200, n_sectors=1,
                     sector_days=27.4, gap_days=1.0,
                     momentum_dump_days=3.1, momentum_dump_len_h=0.25,
                     supersample=3),
    # Ground: one window per night around local midnight, target observed
    # only above min_alt_deg, some nights lost to weather.
    "ground":   dict(kind="ground", cadence_s=60, n_nights=10,
                     night_hours=9.0, latitude_deg=30.0, dec_deg=10.0,
                     min_alt_deg=30.0, meridian_offset_h=None,
                     weather_loss=0.3, supersample=1),
}

# Starting points for noise levels (rough, per exposure).
NOISE_PRESETS = {
    "none":          dict(),
    "tess_bright":   dict(white_ppm=250, red_ppm=50, red_tau_hours=2.0,
                          ramp_ppm=300),
    "tess_mid":      dict(white_ppm=1000, red_ppm=150, red_tau_hours=2.0,
                          ramp_ppm=600),
    "tess_faint":    dict(white_ppm=3000, red_ppm=300, red_tau_hours=2.0,
                          ramp_ppm=1000),
    "ground_1m":     dict(white_ppm=1200, scint_ppm=600, red_ppm=600,
                          red_tau_hours=0.5, extinction_resid=2e-3,
                          night_offset_ppm=500),
    "ground_small":  dict(white_ppm=4000, scint_ppm=1500, red_ppm=1500,
                          red_tau_hours=0.5, extinction_resid=5e-3,
                          night_offset_ppm=1500),
}

STAR_PRESETS = {
    "quiet":       dict(),
    "sunlike":     dict(rot_period=25.0, rot_amp_ppm=300, gran_ppm=60),
    "active_k":    dict(rot_period=8.0, rot_amp_ppm=3000, gran_ppm=150,
                        flare_rate=0.1),
    "m_dwarf":     dict(rot_period=3.0, rot_amp_ppm=8000, gran_ppm=200,
                        flare_rate=1.0, flare_amp_max=0.2),
}

_STAR_DEFAULTS = dict(
    rot_period=None,      # stellar rotation period [d]; None = no spots
    rot_amp_ppm=0.0,      # semi-amplitude of spot modulation
    spot_evol_days=10.0,  # timescale on which spot pattern evolves
    gran_ppm=0.0,         # granulation / short-timescale variability rms
    gran_tau_hours=0.5,   # its correlation time
    flare_rate=0.0,       # mean flares per day (Poisson)
    flare_amp_min=1e-3,   # peak relative amplitude range (power-law dN/dA)
    flare_amp_max=5e-2,
    flare_alpha=2.0,      # dN/dA ∝ A^-alpha
    flare_fwhm_min=2.0,   # flare FWHM range [minutes], log-uniform
    flare_fwhm_max=30.0,
    pulsations=(),        # [(period [d], semi-amplitude [ppm]), ...]
)

_NOISE_DEFAULTS = dict(
    white_ppm=0.0,        # Gaussian white noise per exposure
    red_ppm=0.0,          # correlated (instrumental / atmospheric) noise
    red_tau_hours=1.0,
    scint_ppm=0.0,        # ground: scintillation at airmass 1 (∝ X^1.75)
    extinction_resid=0.0, # ground: residual flux trend per unit airmass
    night_offset_ppm=0.0, # ground: random zero-point per night
    ramp_ppm=0.0,         # space: exponential ramp after each data gap
    ramp_tau_days=0.3,
)


def _ou_process(t, sigma, tau, rng):
    """Ornstein-Uhlenbeck (exponentially correlated) noise on any time grid."""
    x = np.zeros_like(t)
    if sigma <= 0 or t.size == 0:
        return x
    # clip: abutting segments can overlap by ~1e-11 d (a > 1 -> NaN for ever after)
    a = np.exp(-np.clip(np.diff(t), 0, None) / tau)
    s = sigma * np.sqrt(1 - a**2)
    z = rng.normal(size=t.size)
    x[0] = sigma * z[0]
    for k in range(1, t.size):
        x[k] = a[k - 1] * x[k - 1] + s[k - 1] * z[k]
    return x


def flare_profile(t, t_peak, fwhm, amp):
    """Davenport et al. (2014) empirical white-light flare template.
    fwhm in days, amp = peak relative flux increase."""
    x = (t - t_peak) / fwhm
    f = np.zeros_like(t)
    r = (x > -1) & (x <= 0)
    f[r] = (1 + 1.941 * x[r] - 0.175 * x[r]**2 - 2.246 * x[r]**3
            - 1.125 * x[r]**4)
    d = x > 0
    f[d] = 0.6890 * np.exp(-1.600 * x[d]) + 0.3030 * np.exp(-0.2783 * x[d])
    return amp * f


def make_time_grid(instrument="tess", t_start=0.0, rng=None, **kw):
    """
    Build observation times for an instrument preset (see INSTRUMENTS).
    Returns t [d], segment id (orbit or night), airmass (NaN in space), cfg.
    """
    rng = np.random.default_rng(rng)
    cfg = dict(INSTRUMENTS[instrument]) if isinstance(instrument, str) \
        else dict(instrument)
    cfg.update(kw)
    dt = cfg["cadence_s"] / 86400.0

    if cfg["kind"] == "space":
        ts, seg = [], []
        half = (cfg["sector_days"] - cfg["gap_days"]) / 2
        for s in range(cfg["n_sectors"]):
            s0 = t_start + s * cfg["sector_days"]
            for o, o0 in enumerate([s0, s0 + half + cfg["gap_days"]]):
                tt = np.arange(o0, o0 + half, dt)
                ts.append(tt)
                seg.append(np.full(tt.size, 2 * s + o))
        t, seg = np.concatenate(ts), np.concatenate(seg)
        # momentum dumps: short data losses at regular intervals
        md = cfg.get("momentum_dump_days")
        if md:
            ph = np.mod(t - t_start, md)
            keep = ph > cfg["momentum_dump_len_h"] / 24.0
            t, seg = t[keep], seg[keep]
        airmass = np.full(t.size, np.nan)

    elif cfg["kind"] == "ground":
        lat = np.radians(cfg["latitude_deg"])
        dec = np.radians(cfg["dec_deg"])
        mo = cfg["meridian_offset_h"]
        if mo is None:
            mo = rng.uniform(-cfg["night_hours"] / 2, cfg["night_hours"] / 2)
        cfg["meridian_offset_h"] = float(mo)
        ts, seg, am = [], [], []
        for n in range(cfg["n_nights"]):
            if rng.random() < cfg["weather_loss"]:
                continue
            mid = t_start + n + 0.5            # local midnight
            hw = cfg["night_hours"] / 48.0
            tt = np.arange(mid - hw, mid + hw, dt)
            H = 2 * np.pi * (tt - mid - mo / 24.0)   # hour angle [rad]
            sin_alt = (np.sin(lat) * np.sin(dec)
                       + np.cos(lat) * np.cos(dec) * np.cos(H))
            ok = sin_alt > np.sin(np.radians(cfg["min_alt_deg"]))
            ts.append(tt[ok])
            seg.append(np.full(ok.sum(), n))
            am.append(1.0 / sin_alt[ok])
        if not ts:
            raise ValueError("all nights lost (weather_loss / altitude cut)")
        t, seg, airmass = map(np.concatenate, (ts, seg, am))
    else:
        raise ValueError("cfg['kind'] must be 'space' or 'ground'")

    return t, seg.astype(int), airmass, cfg


def stellar_signal(t, rng, **star):
    """Return (stellar multiplicative model, flare part, list of flares)."""
    p = dict(_STAR_DEFAULTS)
    p.update(star)
    s = np.zeros_like(t)

    # rotational modulation by evolving spots: 2 harmonics whose amplitudes
    # wander on spot_evol_days (quasi-periodic, like a QP-GP)
    if p["rot_period"] and p["rot_amp_ppm"] > 0:
        A = p["rot_amp_ppm"] * 1e-6
        a1 = 1 + 0.3 * _ou_process(t, 1.0, p["spot_evol_days"], rng)
        a2 = 0.4 + 0.2 * _ou_process(t, 1.0, p["spot_evol_days"], rng)
        ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
        w = 2 * np.pi / p["rot_period"]
        s += A * (a1 * np.sin(w * t + ph1) + a2 * np.sin(2 * w * t + ph2))

    s += _ou_process(t, p["gran_ppm"] * 1e-6, p["gran_tau_hours"] / 24, rng)

    # coherent pulsations (delta Scuti / gamma Dor / RR Lyr-like)
    for per, amp in p["pulsations"]:
        s += amp * 1e-6 * np.sin(2 * np.pi * t / per
                                 + rng.uniform(0, 2 * np.pi))

    # flares: Poisson in time, power-law amplitudes, log-uniform widths
    flares = []
    fl = np.zeros_like(t)
    if p["flare_rate"] > 0 and t.size:
        span = t.max() - t.min()
        nf = rng.poisson(p["flare_rate"] * span)
        al = p["flare_alpha"]
        lo, hi = p["flare_amp_min"], p["flare_amp_max"]
        u = rng.random(nf)
        if np.isclose(al, 1):
            amps = lo * (hi / lo) ** u
        else:
            amps = (lo**(1 - al) + u * (hi**(1 - al) - lo**(1 - al))) \
                ** (1 / (1 - al))
        fwhm = np.exp(rng.uniform(np.log(p["flare_fwhm_min"]),
                                  np.log(p["flare_fwhm_max"]), nf)) / 1440
        tpk = rng.uniform(t.min(), t.max(), nf)
        for tp, fw, am in zip(tpk, fwhm, amps):
            fl += flare_profile(t, tp, fw, am)
            flares.append(dict(t_peak=float(tp), fwhm_min=float(fw * 1440),
                               amp=float(am)))
    return 1.0 + s + fl, fl, flares


def _place_t0(t0, t, P, rng):
    """Resolve t0 given as a number or as 'begin'/'middle'/'end'/'random'/
    'in_data' (a random observed timestamp: guarantees a sampled transit)."""
    if not isinstance(t0, str):
        return float(t0)
    lo, hi = t.min(), t.max()
    frac = {"begin": 0.1, "middle": 0.5, "end": 0.9}
    if t0 in frac:
        return float(lo + frac[t0] * (hi - lo))
    if t0 == "random":
        return float(rng.uniform(lo, min(hi, lo + P)))
    if t0 == "in_data":
        return float(rng.choice(t))
    raise ValueError(f"unknown t0 option {t0!r}")


def simulate_lightcurve(instrument="tess", planet=None, star=None,
                        noise=None, time=None, t_start=0.0, seed=None,
                        eb=None, **instrument_kw):
    """
    Full mock light curve: sampling + ringed-planet transits + stellar
    activity + systematics + noise.

    instrument : key of INSTRUMENTS ('tess', 'tess_20s', 'tess_ffi',
                 'ground') or a config dict; overrides via **instrument_kw
                 (e.g. n_sectors=2, n_nights=5, cadence_s=30).
    planet     : dict of light_curve() arguments (P, a, inc, rp, r_in, r_out,
                 tau, ring_incl/ring_pa or obliquity/ring_azimuth, u1, u2, e,
                 omega, n_grid). t0 may be a number or 'begin', 'middle',
                 'end', 'random', 'in_data'.  None = no planet.
    star       : dict (see _STAR_DEFAULTS) or STAR_PRESETS key.
    noise      : dict (see _NOISE_DEFAULTS) or NOISE_PRESETS key.
                 Default = no noise at all.
    time       : optional custom time array (overrides instrument sampling).
    seed       : random seed for reproducibility.
    eb         : dict of eb_light_curve() arguments (eclipsing binary in
                 place of the planet; t0 options as for planet).

    Returns a dict with arrays (time, flux, flux_err, transit_model,
    stellar_model, flare_model, systematics_model, red_noise, airmass,
    segment, in_transit) and 'meta' (all parameters, transit times, flares).
    """
    rng = np.random.default_rng(seed)

    if time is not None:
        t = np.sort(np.asarray(time, float))
        seg = np.zeros(t.size, int)
        airmass = np.full(t.size, np.nan)
        cfg = dict(kind="custom", supersample=1,
                   cadence_s=float(np.median(np.diff(t)) * 86400))
    else:
        t, seg, airmass, cfg = make_time_grid(instrument, t_start, rng,
                                              **instrument_kw)

    star = STAR_PRESETS[star] if isinstance(star, str) else (star or {})
    noise = NOISE_PRESETS[noise] if isinstance(noise, str) else (noise or {})
    nz = dict(_NOISE_DEFAULTS)
    nz.update(noise)

    # --- transit --------------------------------------------------------
    transit = np.ones_like(t)
    meta_planet, transit_times = None, []
    if planet is not None:
        pl = dict(planet)
        pl.setdefault("t0", "middle")
        pl["t0"] = _place_t0(pl["t0"], t, pl["P"], rng)
        ss = cfg.get("supersample", 1)
        transit = planet_light_curve(t, rng=rng,
                                     exptime=cfg["cadence_s"] / 86400,
                                     supersample=ss, **pl)
        k = np.arange(np.floor((t.min() - pl["t0"]) / pl["P"]),
                      np.ceil((t.max() - pl["t0"]) / pl["P"]) + 1)
        transit_times = [float(pl["t0"] + kk * pl["P"]) for kk in k
                         if t.min() <= pl["t0"] + kk * pl["P"] <= t.max()]
        meta_planet = pl
    elif eb is not None:
        pl = dict(eb)
        pl.setdefault("t0", "middle")
        pl["t0"] = _place_t0(pl["t0"], t, pl["P"], rng)
        transit = eb_light_curve(t, exptime=cfg["cadence_s"] / 86400,
                                 supersample=cfg.get("supersample", 1), **pl)
        k = np.arange(np.floor((t.min() - pl["t0"]) / pl["P"]),
                      np.ceil((t.max() - pl["t0"]) / pl["P"]) + 1)
        transit_times = [float(pl["t0"] + kk * pl["P"]) for kk in k
                         if t.min() <= pl["t0"] + kk * pl["P"] <= t.max()]
        meta_planet = dict(pl, is_eb=True)

    # --- star -----------------------------------------------------------
    stellar, flare_model, flares = stellar_signal(t, rng, **star)

    # --- systematics ----------------------------------------------------
    sysm = np.ones_like(t)
    if cfg["kind"] == "space" and nz["ramp_ppm"] > 0:
        for sid in np.unique(seg):
            m = seg == sid
            sysm[m] -= nz["ramp_ppm"] * 1e-6 * np.exp(
                -(t[m] - t[m].min()) / nz["ramp_tau_days"])
    if cfg["kind"] == "ground":
        sysm *= 1 - nz["extinction_resid"] * (airmass - 1)
        for sid in np.unique(seg):
            sysm[seg == sid] *= 1 + rng.normal(0, nz["night_offset_ppm"] * 1e-6)

    # --- noise ----------------------------------------------------------
    red = _ou_process(t, nz["red_ppm"] * 1e-6, nz["red_tau_hours"] / 24, rng)
    X = np.where(np.isfinite(airmass), airmass, 1.0)
    sig = 1e-6 * np.sqrt(nz["white_ppm"]**2
                         + (nz["scint_ppm"] * X**1.75 *
                            np.isfinite(airmass))**2)
    white = rng.normal(size=t.size) * sig

    flux = transit * stellar * sysm + red + white

    in_tr = transit < 1.0
    if eb is not None:      # eclipses only, not the ellipsoidal modulation
        kw = {k: v for k, v in meta_planet.items() if k != "is_eb"}
        in_tr = eb_light_curve(t, **dict(kw, ellip_ppm=0.0, refl_ppm=0.0)
                               ) < 1.0 - 1e-9
    Pn = meta_planet["P"] if meta_planet else 1.0
    n_obs = sum(1 for tc in transit_times
                if np.any(in_tr & (np.abs(t - tc) < Pn / 4)))

    meta = dict(instrument=cfg, planet=meta_planet, star=star, noise=nz,
                seed=seed, transit_times=transit_times,
                n_transits_observed=int(n_obs),
                n_flares=len(flares), flares=flares,
                n_points=int(t.size))
    return dict(time=t, flux=flux, flux_err=sig, transit_model=transit,
                stellar_model=stellar, flare_model=flare_model,
                systematics_model=sysm, red_noise=red, airmass=airmass,
                segment=seg, in_transit=in_tr, meta=meta)


_COLUMNS = ["time", "flux", "flux_err", "transit_model", "stellar_model",
            "flare_model", "systematics_model", "red_noise", "airmass",
            "segment", "in_transit"]


def save_lightcurve(sim, path_prefix):
    """Write <prefix>.csv (data + noiseless components) and <prefix>.json."""
    import json
    import os
    d = os.path.dirname(path_prefix)
    if d:
        os.makedirs(d, exist_ok=True)
    arr = np.column_stack([np.asarray(sim[c], float) for c in _COLUMNS])
    fmt = ["%.8f", "%.8f", "%.3e", "%.8f", "%.8f", "%.3e", "%.8f", "%.3e",
           "%.4f", "%d", "%d"]
    np.savetxt(path_prefix + ".csv", arr, delimiter=",", fmt=fmt,
               header=",".join(_COLUMNS), comments="")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(path_prefix + ".json", "w") as f:
        json.dump(_clean(sim["meta"]), f, indent=2)


def save_compact(sim, path_prefix):
    """Write <prefix>.npz (time, flux, flux_err, segment, transit_model) and
    <prefix>.json. ~10x smaller than the CSV; train.py reads both."""
    import os
    d = os.path.dirname(path_prefix)
    if d:
        os.makedirs(d, exist_ok=True)
    err = np.broadcast_to(np.asarray(sim["flux_err"], float),
                          sim["time"].shape)
    np.savez_compressed(path_prefix + ".npz",
                        time=sim["time"].astype(np.float64),
                        flux=sim["flux"].astype(np.float32),
                        flux_err=err.astype(np.float32),
                        segment=sim["segment"].astype(np.int16),
                        transit_model=sim["transit_model"].astype(np.float32))
    _save_json(sim, path_prefix)


def _save_json(sim, path_prefix):
    import json

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(path_prefix + ".json", "w") as f:
        json.dump(_clean(sim["meta"]), f, indent=2)



# Fine labels. Stage 1 of train.py: planet/ringed vs everything else;
# stage 2: ringed vs planet.
CLASS_IDS = {"none": 0, "planet": 1, "ringed": 2, "eb": 3, "beb": 4,
             "flare": 5, "variable": 6}
PLANET_CLASSES = ("planet", "ringed")


def random_planet(rng, ring=None, P_range=(0.5, 15.0), rp_range=(0.03, 0.15),
                  unringed_rp_range=None, b_max=0.9, ring_prob=0.5,
                  rho_star_solar=1.0, t0="in_data",
                  r_in_range=(1.1, 1.8), ring_width_range=(0.3, 1.5),
                  tau_range=(0.1, 3.0), n_grid=200, rho_range=None,
                  ecc_frac=0.0, e_max=0.4, oblate_frac=0.0,
                  oblateness_max=0.15, aligned_ring_frac=0.0,
                  aligned_sigma_deg=15.0, ttv_frac=0.0, ttv_max_min=10.0,
                  spot_frac=0.0):
    """
    Draw a random planet.
    ring : True (always ringed), False (never), None (ringed with ring_prob).
    unringed_rp_range : radius range used for ring-less planets (default =
        rp_range). Making it wider (e.g. up to 0.25) lets ring-less planets
        reach the same depths as ringed ones, so a classifier cannot just
        use "deeper = ringed".
    a/R* follows from P and stellar density: a/R* ≈ 4.2 P[d]^(2/3) ρ^(1/3).
    rho_range : (lo, hi) stellar density [solar], log-uniform; overrides
        rho_star_solar so transit duration does not follow from P alone.
    Nuisance effects (same rates for ringed and ring-less planets, so they
    are look-alikes, not labels):
      ecc_frac    : share with e ~ U(0, e_max), random omega
      oblate_frac : share with flattening ~ U(0, oblateness_max)
      ttv_frac    : share with sinusoidal TTVs, amplitude up to ttv_max_min
      spot_frac   : share whose transits cross star spots (bumps)
    aligned_ring_frac : share of rings near the orbital plane (obliquity
        ~ |N(0, aligned_sigma_deg)|, as expected for close-in planets);
        the rest have isotropic orientations.
    """
    if ring is None:
        ring = rng.random() < ring_prob
    P = float(np.exp(rng.uniform(*np.log(P_range))))
    rho = float(np.exp(rng.uniform(*np.log(rho_range)))) if rho_range \
        else rho_star_solar
    a = 4.206 * P ** (2 / 3) * rho ** (1 / 3)
    e, omega = 0.0, 90.0
    if rng.random() < ecc_frac:
        e, omega = float(rng.uniform(0, e_max)), float(rng.uniform(0, 360))
    # impact parameter at conjunction, allowing for eccentricity
    rfac = (1 - e**2) / (1 + e * np.sin(np.radians(omega)))
    b = rng.uniform(0, b_max)
    rr = rp_range if (ring or unringed_rp_range is None) else unringed_rp_range
    pl = dict(P=P, a=float(a), inc=float(np.degrees(np.arccos(
                  min(1.0, b / (a * rfac))))),
              rp=float(rng.uniform(*rr)),
              u1=float(rng.uniform(0.2, 0.6)), u2=float(rng.uniform(0.0, 0.3)),
              t0=t0, n_grid=n_grid, rho_star=rho)
    if e > 0:
        pl.update(e=e, omega=omega)
    # orientation of ring plane = planet equator (also sets oblateness axis)
    if rng.random() < aligned_ring_frac:
        ob = abs(rng.normal(0, aligned_sigma_deg))
        n = ring_normal_from_orbit(ob, rng.uniform(0, 360), pl["inc"])
        incl, pa = sky_angles_from_normal(n)
        pl["ring_aligned"] = 1
    else:
        incl = float(np.degrees(np.arccos(rng.uniform(0, 1))))
        pa = float(rng.uniform(0, 180))
        pl["ring_aligned"] = 0
    pl.update(ring_incl=float(incl), ring_pa=float(pa))
    if rng.random() < oblate_frac:
        pl["oblateness"] = float(rng.uniform(0, oblateness_max))
    if ring:
        r_in = rng.uniform(*r_in_range)
        pl.update(r_in=float(r_in),
                  r_out=float(r_in + rng.uniform(*ring_width_range)),
                  tau=float(np.exp(rng.uniform(*np.log(tau_range)))))
    else:
        pl.update(tau=0.0)
    if rng.random() < ttv_frac:
        pl["ttv"] = dict(amp_min=float(rng.uniform(1, ttv_max_min)),
                         period_epochs=float(rng.uniform(3, 30)),
                         phase=float(rng.uniform(0, 2 * np.pi)))
    if rng.random() < spot_frac:
        pl["spots"] = dict(per_transit=float(rng.uniform(0.3, 2.0)),
                           contrast=float(rng.uniform(0.1, 0.7)),
                           width_min=float(rng.uniform(5, 40)))
    return pl


_PLANET_EXTRA = ("rho_star", "ring_aligned", "ttv", "spots")


def planet_light_curve(t, rng=None, **pl):
    """light_curve() plus the optional nuisance effects in random_planet:
    TTVs (per-epoch shifts of t0) and star-spot crossings (Gaussian bumps
    inside the transit, amplitude contrast x local depth)."""
    rng = np.random.default_rng(rng)
    ttv, spots = pl.get("ttv"), pl.get("spots")
    kw = {k: v for k, v in pl.items() if k not in _PLANET_EXTRA}
    if not ttv:
        f = light_curve(t, **kw)
    else:
        P, t0 = kw["P"], kw["t0"]
        ep = np.round((t - t0) / P)
        f = np.ones_like(t)
        for k in np.unique(ep):
            m = ep == k
            dt = ttv["amp_min"] / 1440 * np.sin(
                2 * np.pi * k / ttv["period_epochs"] + ttv["phase"])
            f[m] = light_curve(t[m], **dict(kw, t0=t0 + k * P + dt))
    if spots:
        P, t0 = kw["P"], kw["t0"]
        ep = np.round((t - t0) / P)
        for k in np.unique(ep[f < 1]):
            m = (ep == k) & (f < 1)
            for _ in range(rng.poisson(spots["per_transit"])):
                tc = rng.uniform(t[m].min(), t[m].max())
                w = spots["width_min"] / 1440 / 2.355
                bump = spots["contrast"] * rng.uniform(0.3, 1.0) * \
                    np.exp(-0.5 * ((t[m] - tc) / w) ** 2)
                f[m] = f[m] + bump * (1 - f[m])
    return f


def ring_signal_ppm(planet, n_pts=400):
    """Largest flux difference [ppm] between the ringed transit and the same
    planet without its ring. ~0 for edge-on or very thin rings."""
    p = {k: v for k, v in planet.items() if k not in _PLANET_EXTRA}
    if p.get("tau", 0) <= 0:
        return 0.0
    p["t0"] = 0.0
    k = p["rp"] * max(p.get("r_out", 1.0), 1.0)
    half = p["P"] / np.pi * np.arcsin(min(1.0, (1 + k) / p["a"])) * 1.2
    t = np.linspace(-half, half, n_pts)
    f_ring = light_curve(t, **p)
    f_bare = light_curve(t, **dict(p, tau=0.0))
    return float(1e6 * np.max(np.abs(f_ring - f_bare)))


def _class_counts(n, fractions):
    """Turn {'ringed':..,'planet':..,'none':..} (any scale) into exact counts."""
    keys = [k for k in CLASS_IDS if fractions.get(k, 0) > 0]
    unknown = set(fractions) - set(CLASS_IDS)
    if unknown:
        raise ValueError(f"unknown classes {unknown}; use {list(CLASS_IDS)}")
    w = np.array([fractions[k] for k in keys], float)
    w /= w.sum()
    c = np.floor(w * n).astype(int)
    for j in np.argsort(-(w * n - c))[: n - c.sum()]:   # largest remainders
        c[j] += 1
    return dict(zip(keys, c.tolist()))


def random_eb(rng, blended=False, P_range=(0.3, 15.0), t0="in_data",
              blend_depth_ppm=(300.0, 30000.0), n_grid=150):
    """
    Draw a random eclipsing binary.
    blended=False : the target itself is an EB (deep, often V-shaped or
                    grazing, secondary eclipses, ellipsoidal variation).
    blended=True  : background / hierarchical EB whose eclipses are diluted
                    to planet-like depths (log-uniform in blend_depth_ppm).
    """
    P = float(np.exp(rng.uniform(*np.log(P_range))))
    q = rng.uniform(0.15, 1.0)                        # mass ratio
    k = float(np.clip(q ** 0.8 * rng.uniform(0.85, 1.15), 0.1, 1.0))
    sb = float(np.clip(q ** 2.5 * rng.uniform(0.5, 1.5), 0.002, 1.0))
    a = 4.206 * P ** (2 / 3) * (1 + q) ** (1 / 3) * rng.uniform(0.8, 1.2)
    a = max(a, 1.5 * (1 + k))
    # include grazing geometries (b up to ~1 + 0.9 k -> V-shaped eclipses)
    b = rng.uniform(0, 1 + 0.9 * k)
    e = 0.0 if P < 3 or rng.random() < 0.5 else float(rng.uniform(0, 0.3))
    eb = dict(P=P, a=float(a), inc=float(np.degrees(np.arccos(b / a))), k=k,
              sb_ratio=sb, e=e, omega=float(rng.uniform(0, 360)),
              u1=float(rng.uniform(0.2, 0.6)), u2=float(rng.uniform(0.0, 0.3)),
              ellip_ppm=float(1e6 * q / a ** 3 * (1 - (b / a) ** 2)),
              refl_ppm=float(1e6 * 0.3 * sb * k ** 2 / a ** 2),
              t0=t0, n_grid=n_grid)
    depth0 = 1 - eb_light_curve([0.0], **dict(eb, t0=0.0))[0]
    if blended:
        target = np.exp(rng.uniform(*np.log(blend_depth_ppm))) * 1e-6
        eb["dilution"] = float(max(depth0 / target - 1, 0.0))
    else:   # some third light even for on-target EBs
        eb["dilution"] = float(rng.uniform(0, 0.5) if rng.random() < 0.3
                               else 0.0)
    return eb


def random_flare_star(rng):
    """Very active, frequently flaring (mostly M-dwarf-like) star."""
    return dict(rot_period=float(np.exp(rng.uniform(np.log(0.3), np.log(15)))),
                rot_amp_ppm=float(np.exp(rng.uniform(np.log(500),
                                                     np.log(20000)))),
                gran_ppm=float(rng.uniform(50, 300)),
                flare_rate=float(np.exp(rng.uniform(np.log(0.5), np.log(6)))),
                flare_amp_min=1e-3,
                flare_amp_max=float(np.exp(rng.uniform(np.log(0.02),
                                                       np.log(1.0)))),
                flare_alpha=float(rng.uniform(1.6, 2.2)))


def random_variable_star(rng):
    """Pulsator (delta Scuti / gamma Dor-like) or fast spotted rotator."""
    if rng.random() < 0.5:
        n = int(rng.integers(1, 6))
        per = np.exp(rng.uniform(np.log(0.02), np.log(2.0), n))
        amp = np.exp(rng.uniform(np.log(100), np.log(10000), n))
        return dict(pulsations=[(float(p), float(a)) for p, a in
                                zip(per, amp)],
                    gran_ppm=float(rng.uniform(20, 150)))
    return dict(rot_period=float(np.exp(rng.uniform(np.log(0.2), np.log(3)))),
                rot_amp_ppm=float(np.exp(rng.uniform(np.log(1000),
                                                     np.log(30000)))),
                spot_evol_days=float(rng.uniform(1, 10)),
                gran_ppm=float(rng.uniform(50, 300)),
                flare_rate=float(rng.uniform(0, 0.5)))


_CAT_KEYS = ["P", "t0", "a", "inc", "rp", "r_in", "r_out", "tau",
             "ring_incl", "ring_pa", "eb_k", "eb_sb_ratio", "eb_dilution",
             "e", "rho_star", "oblateness", "ring_aligned", "ttv", "spots",
             "base"]


def load_real_base(path):
    """Real TESS light curve (tess_download.py npz): time, normalised
    flux, flux_err, segment."""
    z = np.load(path)
    return (z["time"].astype(float), z["flux"].astype(float),
            z["flux_err"].astype(float), z["segment"].astype(int))


def _make_one(job):
    """Worker: simulate + save one light curve, return its catalog row."""
    import os
    (i, s, label, instrument, out_dir, noise_choices, star_choices,
     planet_kw, min_ring_signal_ppm, max_ring_tries, instrument_kw,
     fmt, base) = job
    r = np.random.default_rng(s)
    pl, eb, rsig = None, None, 0.0
    if label in PLANET_CLASSES:
        ringed = label == "ringed"
        for _ in range(max_ring_tries):
            pl = random_planet(r, ring=ringed, **(planet_kw or {}))
            if not ringed:
                break
            rsig = ring_signal_ppm(pl)
            if not min_ring_signal_ppm or rsig >= min_ring_signal_ppm:
                break
    elif label in ("eb", "beb"):
        eb = random_eb(r, blended=label == "beb",
                       P_range=(planet_kw or {}).get("P_range", (0.3, 15.0)))
    nk = str(r.choice(noise_choices)) if base is None else "real"
    if label == "flare":
        sk, star = "flare", random_flare_star(r)
        if base is not None:            # real star: add the flares only
            star = {k: v for k, v in star.items()
                    if k.startswith("flare")}
    elif label == "variable":
        sk, star = "variable", random_variable_star(r)
    elif base is None:
        sk = str(r.choice(star_choices))
        star = sk
    else:
        sk, star = "real", None
    if base is None:
        sim = simulate_lightcurve(instrument, planet=pl, star=star, noise=nk,
                                  seed=s, eb=eb, **instrument_kw)
    else:
        # inject into a real TESS light curve: real noise, systematics and
        # stellar variability; simulated signal multiplies the real flux
        t, f, err, seg = load_real_base(base)
        sim = simulate_lightcurve(time=t, planet=pl, star=star, noise=None,
                                  seed=s, eb=eb)
        sim["flux"] = f * sim["transit_model"] * sim["stellar_model"]
        sim["flux_err"], sim["segment"] = err, seg
        sim["meta"]["base"] = os.path.basename(base)
    name = f"lc_{i:05d}"
    (save_compact if fmt == "npz" else save_lightcurve)(
        sim, os.path.join(out_dir, name))
    mp = sim["meta"]["planet"] or {}
    ringed = bool(pl is not None and mp.get("tau", 0) > 0)
    row = dict(id=name, label=label, class_id=CLASS_IDS[label], seed=s,
               noise=nk, star=sk, has_planet=int(pl is not None),
               has_ring=int(ringed))
    row.update({k: None for k in _CAT_KEYS})
    for k in ("P", "t0", "a", "inc", "rp", "e", "rho_star", "oblateness",
              "ring_aligned"):
        row[k] = mp.get(k)
    if pl is not None:
        row.update(ttv=int("ttv" in pl), spots=int("spots" in pl))
    if base is not None:
        row["base"] = os.path.splitext(os.path.basename(base))[0]
    if ringed:
        for k in ("r_in", "r_out", "tau", "ring_incl", "ring_pa"):
            row[k] = mp.get(k)
    if eb is not None:
        row.update(eb_k=mp["k"], eb_sb_ratio=mp["sb_ratio"],
                   eb_dilution=mp["dilution"])
    row.update(
        ring_signal_ppm=rsig if ringed else 0.0,
        n_transits_observed=sim["meta"]["n_transits_observed"],
        n_flares=sim["meta"]["n_flares"],
        depth_ppm=float(1e6 * (1 - sim["transit_model"].min())),
        n_points=sim["meta"]["n_points"],
    )
    return row


def generate_dataset(n, out_dir, instrument="tess", seed=0,
                     class_fractions=None,
                     noise_choices=("none", "tess_bright", "tess_faint"),
                     star_choices=("quiet", "sunlike", "active_k", "m_dwarf"),
                     planet_kw=None, min_ring_signal_ppm=None,
                     max_ring_tries=50, n_jobs=1, fmt="csv",
                     real_base=None, **instrument_kw):
    """
    Generate n mock light curves into out_dir (one CSV + JSON each) plus
    catalog.csv with labels.

    class_fractions : share of each class, any scale (normalised), e.g.
        dict(ringed=1.0)                          -> 100% ringed planets
        dict(ringed=40, planet=40, none=20)       -> 40% / 40% / 20%
      classes: 'ringed' (planet with ring), 'planet' (no ring),
               'none' (no planet: star + noise only),
               'eb' (eclipsing binary on target), 'beb' (blended /
               background EB diluted to planet-like depth), 'flare' (strongly
               flaring active star), 'variable' (pulsator / fast rotator).
      Counts are exact; labels are shuffled. Default = equal thirds.
    min_ring_signal_ppm : for ringed planets, redraw the ring (up to
        max_ring_tries) until it changes the transit by at least this many
        ppm compared with the bare planet. Removes nearly invisible
        (edge-on / thin) rings that would only add label noise.
    planet_kw : passed to random_planet (P_range, rp_range,
        unringed_rp_range, tau_range, b_max, t0, ...).
    n_jobs : parallel worker processes (on macOS/Windows call this from
        inside `if __name__ == "__main__":`).
    fmt : 'csv' (all model components) or 'npz' (compact).
    real_base : list of real TESS light curves (tess_download.py npz). If
        given, every signal is injected into one of them (used in turn,
        shuffled) instead of simulated noise / star; noise_choices and
        star_choices are then ignored.
    Extra keywords go to the instrument (n_sectors, n_nights, cadence_s...).
    """
    import csv
    import os
    os.makedirs(out_dir, exist_ok=True)
    counts = _class_counts(n, class_fractions or
                           dict(ringed=1, planet=1, none=1))
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[k] * v for k, v in counts.items()])
    rng.shuffle(labels)
    seeds = rng.integers(2**31, size=n)
    order = rng.permutation(len(real_base)) if real_base else None
    jobs = [(i, int(seeds[i]), str(labels[i]), instrument, out_dir,
             noise_choices, star_choices, planet_kw, min_ring_signal_ppm,
             max_ring_tries, instrument_kw, fmt,
             None if not real_base else
             real_base[order[i % len(order)]]) for i in range(n)]
    print(f"generating {n} light curves: {counts}")

    rows = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs) as ex:
            for k, row in enumerate(ex.map(_make_one, jobs, chunksize=4)):
                rows.append(row)
                if (k + 1) % max(1, n // 20) == 0 or k + 1 == n:
                    print(f"  {k + 1}/{n}")
    else:
        for k, job in enumerate(jobs):
            rows.append(_make_one(job))
            print(f"  [{k + 1}/{n}] {rows[-1]['id']}  {rows[-1]['label']:7s}"
                  f" noise={rows[-1]['noise']} star={rows[-1]['star']}")

    with open(os.path.join(out_dir, "catalog.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_simulation(sim, fname=None, zoom_hours=12):
    """Full light curve, noiseless components and a zoom on each transit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, f = sim["time"], sim["flux"]
    tts = [tc for tc in sim["meta"]["transit_times"]
           if np.any(np.abs(t - tc) < zoom_hours / 48)][:4]
    nz = max(1, len(tts))
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, f, ".", ms=1.5, color="0.4", label="mock data")
    mdl = sim["transit_model"] * sim["stellar_model"] * sim["systematics_model"]
    gap = np.where(np.diff(t) > 5 * np.median(np.diff(t)))[0] + 1
    ax.plot(np.insert(t, gap, np.nan), np.insert(mdl, gap, np.nan), "-",
            lw=0.8, color="C3", label="noiseless model")
    for tc in sim["meta"]["transit_times"]:
        ax.axvline(tc, color="C0", alpha=0.3, lw=0.8)
    ax.set_xlabel("time [d]")
    ax.set_ylabel("relative flux")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"{sim['meta']['instrument'].get('kind')} | "
                 f"{sim['meta']['n_transits_observed']} transits observed | "
                 f"{sim['meta']['n_flares']} flares")
    for j, tc in enumerate(tts):
        a2 = fig.add_subplot(2, nz, nz + j + 1)
        m = np.abs(t - tc) < zoom_hours / 48
        a2.plot((t[m] - tc) * 24, f[m], ".", ms=2, color="0.4")
        a2.plot((t[m] - tc) * 24, (sim["transit_model"] * sim["stellar_model"]
                                   * sim["systematics_model"])[m], "C3-",
                lw=1)
        a2.set_xlabel("hours from mid-transit")
        a2.set_title(f"t = {tc:.3f}", fontsize=9)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
    return fig


def demo_mock_data():
    # Ringed warm giant around an active K dwarf, one TESS sector
    planet = dict(P=6.3, t0="random", a=16.0, inc=89.0, rp=0.10,
                  r_in=1.3, r_out=2.2, tau=0.8, ring_incl=72, ring_pa=25)
    tess = simulate_lightcurve("tess", planet=planet, star="active_k",
                               noise="tess_bright", seed=1)
    save_lightcurve(tess, "mock_tess")
    plot_simulation(tess, "mock_tess.png")

    # Same planet seen from the ground: 1-m class telescope, 8 nights,
    # transit forced onto an observed night
    planet["t0"] = "in_data"
    grd = simulate_lightcurve("ground", planet=planet, star="active_k",
                              noise="ground_1m", seed=2, n_nights=8)
    save_lightcurve(grd, "mock_ground")
    plot_simulation(grd, "mock_ground.png", zoom_hours=8)
    print("saved mock_tess.{csv,json,png} and mock_ground.{csv,json,png}")


if __name__ == "__main__":
    demo_ring_orientations()
    demo_mock_data()
