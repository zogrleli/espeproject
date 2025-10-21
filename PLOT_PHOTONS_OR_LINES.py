# 03_filter_MAD_density_v2.py
from pathlib import Path
import h5py, numpy as np, matplotlib.pyplot as plt
from pyproj import Transformer

H5 = Path(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\DATA\ATL03_20190606081626_10550303_007_01_AOI.h5")
BEAMS = ["gt1l","gt1r","gt2l","gt2r","gt3l","gt3r"]

# --- Softer defaults ---
WINDOW_M   = 80.0          # was 50
Q_LOW      = 0.5           # was 1.0
Q_HIGH     = 3.0           # was 2.0
ITERATE_ONCE = False       # turn on later

DX_BIN       = 20.0        # was 10
DZ_BIN       = 3.0         # was 2
MIN_BIN_COUNT= 3           # was 5

NEI_RX   = 25.0            # was 15
NEI_DZ   = 3.0             # was 2.5
NEI_K    = 2               # was 4

# Optionally disable steps to isolate the culprit
USE_MAD      = True
USE_DENSITY  = True
USE_NEIGHBOR = True

to_utm = Transformer.from_crs("EPSG:4326","EPSG:32620", always_xy=True)

def _nanmedian_mad(a):
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    return med, mad

def robust_line(x, y, max_pairs=20000):
    x = np.asarray(x); y = np.asarray(y)
    n = x.size
    if n < 2: return 0.0, np.nanmedian(y) if n else (0.0, 0.0)
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, size=min(max_pairs, n))
    j = rng.integers(0, n, size=min(max_pairs, n))
    msk = (j != i)
    i, j = i[msk], j[msk]
    if i.size == 0:
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(b)
    slopes = (y[j]-y[i])/(x[j]-x[i])
    m = np.nanmedian(slopes)
    b = np.nanmedian(y - m*x)
    return float(m), float(b)

def alongtrack(E,N):
    d = np.hypot(np.diff(E), np.diff(N))
    return np.concatenate(([0.0], np.cumsum(d)))

def load_beam(f, g):
    base = f.get(f"{g}/heights")
    if base is None or "h_ph" not in base: return None, None
    h = base["h_ph"][...].astype(float)
    if ("easting" in base) and ("northing" in base):
        E = base["easting"][...].astype(float)
        N = base["northing"][...].astype(float)
    else:
        lon = base["lon_ph"][...].astype(float)
        lat = base["lat_ph"][...].astype(float)
        ok = np.isfinite(lon)&np.isfinite(lat)
        E, N = to_utm.transform(lon[ok], lat[ok])
        h = h[ok]
        E, N = np.asarray(E), np.asarray(N)
    ok = np.isfinite(E)&np.isfinite(N)&np.isfinite(h)
    E, N, h = E[ok], N[ok], h[ok]
    if E.size < 5: return None, None
    x = alongtrack(E,N)
    order = np.argsort(x)
    return x[order], h[order]

def mad_gate(x,res,W,qL,qH):
    xmin,xmax = x.min(), x.max()
    nb = max(1, int(np.ceil((xmax-xmin)/W)))
    edges = np.linspace(xmin, xmax, nb+1)
    idx = np.clip(np.searchsorted(edges, x, side="right")-1, 0, nb-1)
    keep = np.zeros_like(res, bool)
    for b in range(nb):
        m = (idx==b)
        if m.sum()<10: continue
        r = res[m]
        med, mad = _nanmedian_mad(r)
        if mad==0: mad = np.nanmedian(np.abs(r-med))+1e-6
        lo = med - (qL/0.6745)*mad
        hi = med + (qH/0.6745)*mad
        keep[m] = (r>=lo)&(r<=hi)
    return keep

def density_filter(x,res,dx,dz,minc):
    xe = np.arange(x.min(), x.max()+dx, dx)
    ze = np.arange(np.nanmin(res), np.nanmax(res)+dz, dz)
    xi = np.clip(np.searchsorted(xe,x,side="right")-1, 0, len(xe)-2)
    zi = np.clip(np.searchsorted(ze,res,side="right")-1, 0, len(ze)-2)
    flat = xi*(len(ze)-1)+zi
    u,c = np.unique(flat, return_counts=True)
    good = set(u[c>=minc])
    return np.array([(xi[k]*(len(ze)-1)+zi[k]) in good for k in range(x.size)])

def neighbor_filter(x,res,rx,dz,k):
    order = np.argsort(x)
    xs, rs = x[order], res[order]
    keep_ord = np.zeros_like(xs, bool)
    j0 = 0
    for i in range(xs.size):
        xi = xs[i]
        while xs[j0] < xi-rx: j0 += 1
        j1 = i
        while j1+1<xs.size and xs[j1+1]<=xi+rx: j1 += 1
        band = (np.abs(rs[j0:j1+1]-rs[i]) <= dz)
        if band.sum()-1 >= k:
            keep_ord[i]=True
    keep = np.zeros_like(keep_ord)
    keep[order]=keep_ord
    return keep

def run(gtx, x, h):
    print(f"\n[{gtx}] start: {x.size:,} photons")
    m,b = robust_line(x,h)
    trend = m*x + b
    res = h - trend

    kept = np.ones_like(x, bool)

    if USE_MAD:
        k1 = mad_gate(x,res,WINDOW_M,Q_LOW,Q_HIGH)
        kept &= k1
        print(f"[{gtx}] after MAD: {kept.sum():,}")

    if USE_DENSITY:
        k2 = density_filter(x[kept], res[kept], DX_BIN, DZ_BIN, MIN_BIN_COUNT)
        tmp = np.zeros_like(kept); tmp[np.flatnonzero(kept)[k2]] = True
        kept = tmp
        print(f"[{gtx}] after density: {kept.sum():,}")

    if USE_NEIGHBOR:
        k3 = neighbor_filter(x[kept], res[kept], NEI_RX, NEI_DZ, NEI_K)
        tmp = np.zeros_like(kept); tmp[np.flatnonzero(kept)[k3]] = True
        kept = tmp
        print(f"[{gtx}] after neighbor: {kept.sum():,}")

    # If still zero, auto-relax once
    if kept.sum()==0:
        print(f"[{gtx}] nothing kept → auto-relax thresholds and retry MAD only")
        k1 = mad_gate(x,res, max(WINDOW_M,120.0), 0.3, 4.0)
        kept = k1
        print(f"[{gtx}] after relaxed MAD: {kept.sum():,}")

    # Plot
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,6),sharex=True,
                               gridspec_kw={"height_ratios":[2,1]})
    ax1.scatter(x,h,s=1,color="#bdbdbd",label="all (AOI path)")
    ax1.scatter(x[kept],h[kept],s=1,color="crimson",label="kept")
    ax1.plot(x,trend,color="k",lw=1,label="robust trend")
    ax1.set_ylabel("Photon height h_ph (m)")
    ax1.set_title(f"{gtx} — filtered surface   kept {kept.sum():,}/{x.size:,} ({kept.sum()/x.size:.1%})")
    ax1.legend(loc="upper right",fontsize=9)

    ax2.axhline(0,color="k",lw=1)
    ax2.scatter(x[kept], (h-trend)[kept], s=1, color="crimson")
    ax2.set_ylabel("Detrended residual (m)")
    ax2.set_xlabel("Along-track distance (m)")
    plt.tight_layout(); plt.show()

def main():
    assert H5.exists(), f"Missing: {H5}"
    with h5py.File(H5,"r") as f:
        for g in BEAMS:
            xh = load_beam(f,g)
            if xh[0] is None: continue
            run(g,*xh)

if __name__=="__main__":
    main()
