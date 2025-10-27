# kriging_segments20m.py
# ATL03 → AOI + land-ice conf>=2 → MAD filter (Van Tiggelen eq.7) → 1D kriging (Gaussian) clipped
# → High-pass (λ) → Roughness per 20 m segments: H = 2 * std(z_highpass)

from pathlib import Path
import h5py, numpy as np, matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer

# ================== ΡΥΘΜΙΣΕΙΣ ==================
H5_FILE = Path(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\DATA\ATL03_20190606081626_10550303_007_01.h5")

# AOI (UTM20N, μέτρα)
E_MIN, E_MAX = 359000, 368500
N_MIN, N_MAX = 8480000, 8489000

# ATL03 επιλογή
CONF_MIN = 2                  # κρατάμε land-ice photons με conf >= 2

# MAD φίλτρο (Van Tiggelen eq.7) πάνω σε along-track residuals
MAD_WINDOW_M = 50.0
Q_LOW, Q_HIGH = 1.0, 2.0

# Kriging (Gaussian 1D κατά μήκος Easting)
RANGE_M = 30.0                # Gaussian range R (15–40 m ταιριάζει καλά)
MAX_NEIGHBORS = 100
MIN_NEIGHBORS = 3
GRID_STEP = 1.0               # βήμα Easting για το kriging grid
EDGE_SUPPORT_R = RANGE_M      # για “two-sided support” μάσκα άκρων

# High-pass & roughness
HP_LAMBDA = 35.0              # μήκος κύματος low-pass (m) για high-pass
SEG_LEN = 20.0                # μήκος segment (m) — ΖΗΤΗΘΗΚΕ
MIN_PTS_SEG = 10              # ελάχιστα έγκυρα σημεία στο segment για H
MIN_PTS_HP = 30               # ελάχιστα έγκυρα σημεία συνολικά για να βγάλουμε HP

# Έξοδος
OUTDIR = H5_FILE.parent / "kriging_segments20m"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Μετατροπή σε UTM20N
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)
# =================================================


# ----------------- βοηθητικά -----------------
def robust_line(x, y):
    """Γρήγορη/σταθερή robust ευθεία (median-of-slopes)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 5:
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)
    rng = np.random.default_rng(0)
    m = min(20000, x.size)
    i = rng.integers(0, x.size, size=m)
    j = rng.integers(0, x.size, size=m)
    mask = i != j
    i, j = i[mask], j[mask]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    a = np.nanmedian(slopes)
    b = np.nanmedian(y - a * x)
    return float(a), float(b)

def alongtrack_distance(E, N):
    d = np.hypot(np.diff(E), np.diff(N))
    return np.concatenate(([0.0], np.cumsum(d)))

def mad_gate(x, z, W, qL, qH):
    """MAD gate σε sliding bins κατά μήκος του x (σε μέτρα)."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    idx = np.argsort(x); xs, zs = x[idx], z[idx]
    keep_ord = np.zeros_like(zs, bool)

    nb = max(1, int(np.ceil((xs.max() - xs.min()) / W)))
    edges = np.linspace(xs.min(), xs.max(), nb + 1)
    bi = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, nb - 1)

    for b in range(nb):
        m = (bi == b)
        if m.sum() < 15:
            continue
        r = zs[m]
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad == 0:
            mad = np.median(np.abs(r - med)) + 1e-9
        lo = med - (qL / 0.6745) * mad
        hi = med + (qH / 0.6745) * mad
        keep_ord[m] = (r >= lo) & (r <= hi)

    keep = np.zeros_like(keep_ord)
    keep[idx] = keep_ord
    return keep

def gaussian_cov(h, sill, R):
    return sill * np.exp(-0.5 * (h / R) ** 2)

def ok1d(xo, zo, xg, R=30.0, max_k=100, min_pts=3):
    """1D Ordinary Kriging κατά μήκος Easting (Gaussian)."""
    xo = np.asarray(xo, float); zo = np.asarray(zo, float); xg = np.asarray(xg, float)
    order = np.argsort(xo); xo, zo = xo[order], zo[order]
    sill = max(float(np.var(zo)), 1e-4)
    out = np.full_like(xg, np.nan, dtype=float)

    for i, xc in enumerate(xg):
        z_est = np.nan
        for rad in (1.25*R, 2.5*R, 5.0*R):
            left = np.searchsorted(xo, xc - rad, side="left")
            right = np.searchsorted(xo, xc + rad, side="right")
            idx = np.arange(left, right)
            if idx.size >= min_pts:
                dists = np.abs(xo[idx] - xc)
                sub = idx[np.argsort(dists)[:max_k]]
                X = xo[sub]; Z = zo[sub]
                n = X.size
                C = gaussian_cov(np.abs(X[:, None] - X[None, :]), sill, R)
                C.flat[::n+1] += 1e-6   # μικρό nugget για ευστάθεια
                A = np.block([[C, np.ones((n, 1))],
                              [np.ones((1, n)), np.zeros((1, 1))]])
                rhs = np.r_[gaussian_cov(np.abs(X - xc), sill, R), 1.0]
                try:
                    wlam = np.linalg.solve(A, rhs)
                    w = wlam[:n]
                    z_est = float(w @ Z)
                except np.linalg.LinAlgError:
                    z_est = np.nan
                if np.isfinite(z_est):
                    break
        out[i] = z_est
    return out

def two_sided_support(x_data, x_grid, side_radius):
    """True όπου υπάρχει τουλάχιστον 1 σημείο ≤side_radius αριστερά ΚΑΙ δεξιά (όχι extrapolation άκρων)."""
    xd = np.sort(np.asarray(x_data, float))
    xg = np.asarray(x_grid, float)
    mask = np.zeros_like(xg, bool)

    for i, x in enumerate(xg):
        j = np.searchsorted(xd, x)
        has_left = (j > 0) and (x - xd[max(0, j-1)] <= side_radius)
        has_right = (j < xd.size) and (xd[min(xd.size-1, j)] - x <= side_radius)
        mask[i] = bool(has_left and has_right)
    return mask

def highpass_profile(x, z, lambda_c):
    """High-pass = z - moving-average (παράθυρο ~λ). Edge-safe convolution με NaNs."""
    z = np.asarray(z, float)
    if np.count_nonzero(np.isfinite(z)) < MIN_PTS_HP:
        return np.full_like(z, np.nan)

    L = int(round(lambda_c))
    if L < 3: L = 3
    if L % 2 == 0: L += 1

    zmir = np.r_[z[::-1], z, z[::-1]]
    w = np.isfinite(zmir).astype(float)
    z0 = np.where(np.isfinite(zmir), zmir, 0.0)

    k = np.ones(L)
    num = np.convolve(z0, k, mode="same")
    den = np.convolve(w,  k, mode="same")
    low = num / np.maximum(den, 1e-12); low[den < 1] = np.nan
    low = low[len(z):2*len(z)]
    return z - low

def segment_stats_20m(x, z_hp, seg_len=20.0, min_pts=10):
    """Σπάει σε συνεχόμενα segments 20 m και δίνει (center, N, sigma, H=2σ)."""
    x = np.asarray(x, float); z_hp = np.asarray(z_hp, float)
    valid = np.isfinite(x) & np.isfinite(z_hp)
    x, z_hp = x[valid], z_hp[valid]
    if x.size < min_pts:
        return np.array([]), np.array([]), np.array([]), np.array([])

    xmin, xmax = float(x.min()), float(x.max())
    edges = np.arange(xmin, xmax + seg_len, seg_len)
    centers = 0.5 * (edges[:-1] + edges[1:])
    N = np.zeros(centers.size, int)
    sigma = np.full(centers.size, np.nan)
    H = np.full(centers.size, np.nan)

    idx = np.clip(np.searchsorted(edges, x, side="right")-1, 0, edges.size-2)
    for k in range(centers.size):
        m = (idx == k)
        if m.sum() >= min_pts:
            zz = z_hp[m] - np.nanmean(z_hp[m])
            s = float(np.nanstd(zz))
            sigma[k] = s
            H[k] = 2.0 * s
            N[k] = m.sum()
    return centers, N, sigma, H

def get_landice_conf(hgrp, npts):
    """Διάβασμα land_ice_conf από signal_conf_ph (σωστό indexing)."""
    if "signal_conf_ph" not in hgrp:
        return np.full(npts, 3, dtype=np.int16)
    sc = np.asarray(hgrp["signal_conf_ph"])
    if sc.ndim == 2 and 5 in sc.shape:
        lic = sc[3, :] if sc.shape[0] == 5 else sc[:, 3]
    else:
        lic = np.full(npts, 3, dtype=np.int16)
    out = np.full(npts, 3, dtype=np.int16)
    m = min(npts, lic.size)
    out[:m] = lic[:m]
    return out

# -------------- main --------------
def main():
    all_rows = []

    with h5py.File(H5_FILE, "r") as f:
        beams = [k for k in f.keys() if k.startswith("gt")]
        if not beams:
            raise SystemExit("No gt* beams in file")

        for gtx in beams:
            base = f.get(f"{gtx}/heights") or f.get(gtx)
            if base is None or "h_ph" not in base:
                print(f"[skip] {gtx}: no heights")
                continue

            # Συντεταγμένες
            if "easting" in base and "northing" in base:
                E = np.asarray(base["easting"])
                N = np.asarray(base["northing"])
            elif "lon_ph" in base and "lat_ph" in base:
                lon = np.asarray(base["lon_ph"]); lat = np.asarray(base["lat_ph"])
                E, N = to_utm.transform(lon, lat); E, N = np.asarray(E), np.asarray(N)
            else:
                print(f"[skip] {gtx}: missing coords")
                continue

            H = np.asarray(base["h_ph"])
            T = np.asarray(base["delta_time"]) if "delta_time" in base else np.arange(H.size, dtype=float)
            n = min(E.size, N.size, H.size, T.size)
            E, N, H, T = E[:n], N[:n], H[:n], T[:n]

            conf = get_landice_conf(base, n)

            # AOI + confidence
            ok_aoi = (E >= E_MIN) & (E <= E_MAX) & (N >= N_MIN) & (N <= N_MAX)
            ok_conf = conf >= CONF_MIN
            ok = ok_aoi & ok_conf & np.isfinite(E) & np.isfinite(N) & np.isfinite(H)
            if ok.sum() < 50:
                print(f"[skip] {gtx}: too few in AOI after conf≥{CONF_MIN}")
                continue

            E = E[ok]; N = N[ok]; H = H[ok]; T = T[ok]

            # along-track (για detrend και MAD)
            order = np.argsort(T)
            Eo, No, Ho = E[order], N[order], H[order]
            s = alongtrack_distance(Eo, No)
            a, b = robust_line(s, Ho)
            res = Ho - (a * s + b)
            keep = mad_gate(s, res, MAD_WINDOW_M, Q_LOW, Q_HIGH)

            E_keep = Eo[keep]; H_keep = Ho[keep]
            if E_keep.size < 30:
                print(f"[skip] {gtx}: too few after MAD")
                continue

            # ---- 1D kriging (μόνο μέσα στα δεδομένα) ----
            e_min = E_keep.min(); e_max = E_keep.max()
            xg = np.arange(e_min, e_max + GRID_STEP, GRID_STEP)
            zg = ok1d(E_keep, H_keep, xg, R=RANGE_M, max_k=MAX_NEIGHBORS, min_pts=MIN_NEIGHBORS)

            # κόψιμο άκρων: two-sided support
            support = two_sided_support(E_keep, xg, side_radius=EDGE_SUPPORT_R)
            zg_clip = np.where(support, zg, np.nan)

            # ---- High-pass ----
            zhp = highpass_profile(xg, zg_clip, HP_LAMBDA)

            # ---- Segments 20 m → H = 2σ ----
            centers, Nseg, sigma, Hseg = segment_stats_20m(xg, zhp, seg_len=SEG_LEN, min_pts=MIN_PTS_SEG)

            # ====== PLOT (3 πάνελ) ======
            fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
            # 1) Kriged
            axs[0].plot(xg, zg_clip, lw=1.6)
            axs[0].set_ylabel("z (m)")
            axs[0].set_title(f"{gtx} — Kriged profile (clipped)")

            # 2) High-pass
            axs[1].plot(xg, zhp, lw=1.2)
            axs[1].axhline(0, color="k", lw=0.6, alpha=0.4)
            axs[1].set_ylabel("high-pass z (m)")
            axs[1].set_title(f"High-pass (λ = {HP_LAMBDA:.0f} m)")

            # 3) H ανά 20 m
            axs[2].plot(centers, Hseg, ".-", lw=1.0, ms=5)
            axs[2].set_ylabel("H (m)")
            axs[2].set_xlabel("Easting (m, UTM20N)")
            axs[2].set_title(f"Roughness per {SEG_LEN:.0f} m (H = 2·σ), min pts {MIN_PTS_SEG}")

            axs[0].grid(alpha=0.25, ls=":")
            axs[1].grid(alpha=0.25, ls=":")
            axs[2].grid(alpha=0.25, ls=":")

            plt.tight_layout()
            out_png = OUTDIR / f"{H5_FILE.stem}_{gtx}_krig_hp_H_{int(SEG_LEN)}m.png"
            plt.savefig(out_png, dpi=220)
            plt.close(fig)

            # ====== CSV ανά beam ======
            rows = []
            for cx, npt, sgm, hh in zip(centers, Nseg, sigma, Hseg):
                if np.isfinite(hh):
                    rows.append({
                        "beam": gtx,
                        "E_center_m": float(cx),
                        "seg_len_m": float(SEG_LEN),
                        "N_pts": int(npt),
                        "sigma_m": float(sgm),
                        "H_corr_m": float(hh)
                    })
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(OUTDIR / f"{H5_FILE.stem}_{gtx}_H_segments_{int(SEG_LEN)}m.csv", index=False)
                all_rows.extend(rows)
                print(f"[OK] {gtx}: saved plot + {len(rows)} segments")

    # Συγκεντρωτικό CSV
    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUTDIR / "ALL_Roughness_20m.csv", index=False)
        print(f"[DONE] Wrote ALL_Roughness_20m.csv in {OUTDIR}")
    else:
        print("[NOTE] No segments produced (insufficient valid data).")

if __name__ == "__main__":
    main()
