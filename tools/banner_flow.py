#!/usr/bin/env python
"""2D lattice Boltzmann (D2Q9, BGK) simulation of flow past a NACA 4418 airfoil, for the README banner.

    python tools/banner_flow.py --output build/banner_flow.npz

Uniform inflow on the left, free-stream rows at the top and bottom, zero-gradient outflow on the right, and
bounce-back on the airfoil. The run saves several late snapshots of (u, v) and the solid mask, so the banner can
draw streamlines and vorticity from one consistent, fully developed flow field. CPU only, a few minutes with torch.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as MPath

NX, NY = 1150, 360
CHORD, AOA_DEG = 100, 15.0
LE = (432.0, 192.0)          # leading-edge position in lattice cells (x, y), y measured upwards
U0, RE = 0.08, 500.0
C = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
W = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]


def naca4(m=0.04, p=0.4, t=0.18, n=400):
    """Closed outline of a NACA 4-digit section with unit chord, cosine-spaced."""
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])


def airfoil_outline():
    x, y = naca4()
    a = np.deg2rad(AOA_DEG)                       # nose up for flow in +x: rotate clockwise
    xr = x * np.cos(a) + y * np.sin(a)
    yr = -x * np.sin(a) + y * np.cos(a)
    return LE[0] + CHORD * xr, LE[1] + CHORD * yr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--snap-every", type=int, default=400)
    ap.add_argument("--snaps", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dt = torch.float32

    nu = U0 * CHORD / RE
    tau = 3 * nu + 0.5
    print(f"Re={RE:.0f} nu={nu:.4f} tau={tau:.4f} Ma={U0 * np.sqrt(3):.3f}", flush=True)

    ox, oy = airfoil_outline()
    yy, xx = np.mgrid[0:NY, 0:NX]
    solid_np = MPath(np.c_[ox, oy]).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(NY, NX)
    solid = torch.from_numpy(solid_np)

    cx = torch.tensor([c[0] for c in C], dtype=dt).view(9, 1, 1)
    cy = torch.tensor([c[1] for c in C], dtype=dt).view(9, 1, 1)
    w = torch.tensor(W, dtype=dt).view(9, 1, 1)

    def feq(rho, ux, uy):
        cu = 3 * (cx * ux + cy * uy)
        return w * rho * (1 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy))

    ones = torch.ones(1, NY, NX, dtype=dt)
    f = feq(ones, U0 * ones, 0 * ones)
    f_in = feq(torch.ones(1, NY, 1, dtype=dt), U0 * torch.ones(1, NY, 1, dtype=dt), torch.zeros(1, NY, 1, dtype=dt))
    f_tb = feq(torch.ones(1, 1, NX, dtype=dt), U0 * torch.ones(1, 1, NX, dtype=dt), torch.zeros(1, 1, NX, dtype=dt))
    omega = 1.0 / tau
    snaps, t0 = [], time.time()
    first_snap = args.steps - (args.snaps - 1) * args.snap_every

    for step in range(1, args.steps + 1):
        rho = f.sum(0, keepdim=True)
        ux = (f * cx).sum(0, keepdim=True) / rho
        uy = (f * cy).sum(0, keepdim=True) / rho
        f = f + omega * (feq(rho, ux, uy) - f)
        f = torch.stack([torch.roll(f[i], shifts=(C[i][1], C[i][0]), dims=(0, 1)) for i in range(9)])
        fs = f[:, solid]
        f[:, solid] = fs[OPP]                      # full-way bounce-back on the airfoil
        f[:, :, :1] = f_in                         # uniform inflow
        f[:, :, -1:] = f[:, :, -2:-1]              # zero-gradient outflow
        f[:, :1, :] = f_tb                         # free stream far above and below
        f[:, -1:, :] = f_tb
        if step % 2000 == 0:
            umax = float(torch.sqrt(ux * ux + uy * uy).max())
            print(f"step {step:6d}  |u|max={umax:.4f}  {time.time() - t0:6.1f}s", flush=True)
            if not np.isfinite(umax) or umax > 0.5:
                raise SystemExit("unstable, increase tau (lower Re) or lower U0")
        if step >= first_snap and (step - first_snap) % args.snap_every == 0:
            u = ux[0].clone(); v = uy[0].clone()
            u[solid] = 0; v[solid] = 0
            snaps.append(np.stack([u.numpy(), v.numpy()]))
            print(f"snapshot at step {step}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, uv=np.stack(snaps).astype(np.float32), solid=solid_np,
                        outline=np.c_[ox, oy], U0=U0, Re=RE, chord=CHORD, aoa=AOA_DEG)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
