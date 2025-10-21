# ICESat2_Shepard_Roughness.py
# Υπολογισμός τραχύτητας (ν) σύμφωνα με Shepard et al. (2001)
# Από το original ATL03 αρχείο: land-ice photons (conf ≥ 2), AOI, dx=0.5 m, segments 20 m

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer

# === ΡΥΘΜΙΣΕΙΣ ===============================================================
H5_FILE = Path(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\DATA\ATL03_20190606081626_10550303_007_01.h5")

# Περιοχή ενδιαφέροντος (AOI, σε μέτρα UTM20N)
E_MIN, E_MAX = 359000, 368500
N_MIN, N_MAX = 8480000, 8489000

# Παράμετροι Shepard
DX = 0.5             # Resample step (m)
SEGMENT_M = 20.0     # Segment length (m)
LAGS = np.array([0.5, 1, 2, 5, 10])   # Lag distances (m)
MIN_PTS = 5          # Ελάχιστα σημεία για RMS υπολογισμό
CONF_MIN = 2         # Ελάχιστο land_ice_conf για αποδοχή
OUTDIR = H5_FILE.parent / "Shepard_Roughness"
OUTDIR.mkdir(exist_ok=True, parents=True)

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)
# ==============================================================================

# --- helpers ------------------------------------------------------------------
def get_landice_conf(hgrp, npts):
    """Επιστρέφει το land_ice_conf από signal_conf_ph (σωστό indexing)."""
    if "signal_conf_ph" not in hgrp:
        return np.full(npts, 3, dtype=np.int16)
    sc = np.asarray(hgrp["signal_conf_ph"])
    if sc.ndim == 2 and 5 in sc.shape:
        landice = sc[3, :] if sc.shape[0] == 5 else sc[:, 3]
    else:
        landice = np.full(npts, 3, dtype=np.int16)
    n = min(npts, landice.size)
    out = np.full(npts, 3, dtype=np.int16)
    out[:n] = landice[:n]
    return out

def linear_detrend(x, z):
    """Αφαιρεί γραμμική τάση z = a*x + b."""
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, z, rcond=None)[0]
    return z - (a * x + b)

def rms_diff(z, k):
    """Υπολογίζει RMS διαφορών για offset k."""
    if k <= 0 or z.size <= k:
        return np.nan
    diff = z[k:] - z[:-k]
    return np.sqrt(np.mean(diff**2))

def uniform_profile(E, Z, dx=0.5):
    """Resample σε ομοιόμορφο πλέγμα dx."""
    order = np.argsort(E)
    E, Z = E[order], Z[order]
    edges = np.arange(E.min(), E.max() + dx, dx)
    centers = (edges[:-1] + edges[1:]) / 2
    z_median = np.full_like(centers, np.nan)
    idx = np.digitize(E, edges) - 1
    for i in range(len(centers)):
        pts = Z[idx == i]
        if len(pts) >= 3:
            z_median[i] = np.median(pts)
    valid = ~np.isnan(z_median)
    return centers[valid], z_median[valid]

def split_segments(E, seg_len=20.0):
    """Σπάει το προφίλ σε segments σταθερού μήκους."""
    Emin, Emax = E.min(), E.max()
    edges = np.arange(Emin, Emax, seg_len)
    segments = []
    for i in range(len(edges) - 1):
        mask = (E >= edges[i]) & (E < edges[i + 1])
        if mask.sum() > 10:
            segments.append(mask)
    return segments

# --- main ---------------------------------------------------------------------
with h5py.File(H5_FILE, "r") as f:
    beams = [k for k in f.keys() if k.startswith("gt")]
    if not beams:
        raise SystemExit("No gt* beams found in file.")

    all_results = []

    for gtx in beams:
        base = f.get(f"{gtx}/heights") or f.get(gtx)
        if base is None or "h_ph" not in base:
            print(f"[skip] {gtx}: no height data.")
            continue

        # Συντεταγμένες και ύψος
        if "easting" in base and "northing" in base:
            E = np.asarray(base["easting"])
            N = np.asarray(base["northing"])
        elif "lon_ph" in base and "lat_ph" in base:
            lon = np.asarray(base["lon_ph"])
            lat = np.asarray(base["lat_ph"])
            E, N = to_utm.transform(lon, lat)
            E, N = np.asarray(E), np.asarray(N)
        else:
            print(f"[skip] {gtx}: missing coordinates.")
            continue

        H = np.asarray(base["h_ph"])
        npts = min(len(E), len(N), len(H))
        E, N, H = E[:npts], N[:npts], H[:npts]
        conf = get_landice_conf(base, npts)

        # Φιλτράρισμα
        ok = (E >= E_MIN) & (E <= E_MAX) & (N >= N_MIN) & (N <= N_MAX) & (conf >= CONF_MIN)
        E, N, H = E[ok], N[ok], H[ok]

        if len(H) < 30:
            print(f"[skip] {gtx}: too few photons in AOI.")
            continue

        # Επαναδειγματοληψία
        Eu, Hu = uniform_profile(E, H, dx=DX)
        if len(Hu) < 30:
            print(f"[skip] {gtx}: too few after resampling.")
            continue

        # Σπάσιμο σε segments 20 m
        seg_masks = split_segments(Eu, seg_len=SEGMENT_M)

        fig, axes = plt.subplots(len(seg_masks), 2, figsize=(10, 3.0*len(seg_masks)), squeeze=False)
        beam_rows = []

        for si, mask in enumerate(seg_masks):
            Es, Zs = Eu[mask], Hu[mask]
            if len(Zs) < 20:
                continue
            Zs_dt = linear_detrend(Es, Zs)
            seg_rows = []

            for lag in LAGS:
                k = int(round(lag / DX))
                nu = rms_diff(Zs_dt, k)
                seg_rows.append({"beam": gtx, "segment": si, "lag_m": lag, "nu_m": nu})

            beam_rows.extend(seg_rows)

            # Plot segment
            axp, axr = axes[si]
            axp.plot(Es, Zs, ".", ms=1.5, color="gray", alpha=0.5)
            axp.plot(Es, Zs_dt + np.mean(Zs), "-", lw=1.0, color="#d62728")
            axp.set_xlabel("Easting (m)")
            axp.set_ylabel("Height (m)")
            axp.set_title(f"{gtx} — segment {si} (≈{Es.max()-Es.min():.1f} m)")

            axr.plot(LAGS, [r["nu_m"] for r in seg_rows], "o-", lw=1.2)
            axr.set_xscale("log"); axr.set_yscale("log")
            axr.set_xlabel("Δx (m)")
            axr.set_ylabel("ν(Δx) (m)")
            axr.grid(True, ls=":")

        # αποθήκευση
        if beam_rows:
            out_csv = OUTDIR / f"{gtx}_roughness.csv"
            pd.DataFrame(beam_rows).to_csv(out_csv, index=False)
            out_png = OUTDIR / f"{gtx}_roughness_panels.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            print(f"[OK] {gtx}: wrote {out_csv.name}, {out_png.name}")
            all_results.extend(beam_rows)
        else:
            plt.close(fig)
            print(f"[skip] {gtx}: no valid segments.")

    # Συνολικό αρχείο
    if all_results:
        out_all = OUTDIR / "ALL_Roughness.csv"
        pd.DataFrame(all_results).to_csv(out_all, index=False)
        print(f"[DONE] saved combined {out_all.name}")
