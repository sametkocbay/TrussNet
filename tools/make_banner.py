#!/usr/bin/env python
"""README banner from a simulated flow snapshot (see tools/banner_flow.py).

    python tools/banner_flow.py --output build/banner_flow.npz
    python tools/make_banner.py --flow build/banner_flow.npz --snapshot 4 --output docs/assets/banner.png

Streamlines are iso-lines of the stream function of the snapshot, so equal spacing means equal volume flux, and
the airfoil surface is itself a streamline. The shading is the vorticity of the same snapshot. Nothing is drawn
by hand except the airfoil outline, which is the simulated geometry. Text is set in JetBrains Mono, found in
TRUSSNET_FONT_DIR or installed system-wide, with DejaVu Sans Mono as the fallback.
"""
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

W, H, DPI = 1600, 460, 100
BG, INK, MUTED, COPPER, ICE, FOIL = "#0d1620", "#eef2f6", "#8fa1b3", "#d9824b", "#5fa8d3", "#22364a"
FONT_DIR = Path(os.environ.get("TRUSSNET_FONT_DIR", Path.home() / ".local/share/fonts/JetBrainsMono-2.304/fonts/ttf"))
X0, X1 = 0.0, 920.0                      # visible window in lattice cells
YC = 180.0


def banner(flow, out, snap, spacing, lim):
    fm = font_manager.fontManager
    for name in ("JetBrainsMono-ExtraBold.ttf", "JetBrainsMono-Regular.ttf", "JetBrainsMono-Medium.ttf"):
        if (FONT_DIR / name).is_file():
            fm.addfont(str(FONT_DIR / name))
    mono = "JetBrains Mono" if any(f.name == "JetBrains Mono" for f in fm.ttflist) else "DejaVu Sans Mono"
    d = np.load(flow)
    u, v = d["uv"][snap].astype(np.float64)
    solid, outline, U0, chord = d["solid"], d["outline"], float(d["U0"]), float(d["chord"])
    ny, nx = u.shape

    # stream function, psi = int u dy - int v dx, integrated from the bottom-left corner
    psi = np.cumsum(u, axis=0) - u[:1] / 2
    psi -= np.cumsum(v[0])[None, :]
    psi /= U0                              # in cells of free-stream flux
    vort = np.gradient(v, axis=1) - np.gradient(u, axis=0)
    vort = vort * chord / U0
    # fill the solid nodes with the nearest fluid value, so the shading and streamlines run right up to the smooth
    # outline drawn on top, instead of stopping at the lattice staircase
    yy, xx = np.mgrid[0:ny, 0:nx]
    wall = binary_dilation(solid, iterations=1)
    _, (iy, ix) = distance_transform_edt(wall, return_indices=True)
    vort = gaussian_filter(np.where(wall, vort[iy, ix], vort), 1.0)

    hh = (X1 - X0) * H / W
    y0, y1 = YC - hh / 2, YC + hh / 2
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off(); ax.set_facecolor(BG)

    cmap = LinearSegmentedColormap.from_list("vort", [ICE, BG, COPPER])
    ax.imshow(np.clip(vort, -lim, lim), extent=(-0.5, nx - 0.5, -0.5, ny - 0.5), origin="lower", cmap=cmap,
              vmin=-lim, vmax=lim, alpha=0.85, interpolation="bicubic", aspect="auto", zorder=1)
    body = float(np.median(psi[solid]))
    levels = body + spacing * np.arange(-60, 61)
    ax.contour(xx, yy, psi, levels=levels, colors=[INK], linewidths=0.8, alpha=0.34,
               linestyles="solid", zorder=2)
    ax.fill(outline[:, 0], outline[:, 1], color=FOIL, ec=FOIL, lw=4.0, zorder=3)
    ax.plot(outline[:, 0], outline[:, 1], color=INK, lw=1.3, zorder=3.5)

    fade = np.clip((np.linspace(0, 1, 600) - 0.12) / 0.32, 0, 1)[None, :] ** 1.6
    rgba = np.zeros((1, 600, 4)); rgba[..., :3] = to_rgb(BG); rgba[..., 3] = 0.92 * (1 - fade)
    ax.imshow(rgba, extent=(X0, X1, y0, y1), aspect="auto", zorder=4)

    tx = fig.transFigure
    ax.text(0.045, 0.60, "TrussNet", transform=tx, fontsize=76, color=INK, family=mono,
            weight="extra bold", va="center", zorder=5)
    ax.text(0.048, 0.385, "Sim-to-real forecasting of turbulent airfoil wakes", transform=tx, fontsize=17,
            color=MUTED, family=mono, va="center", zorder=5)
    ax.text(0.048, 0.25, "3rd place  ·  NeurIPS 2026 RealPDE  ·  Track 1", transform=tx, fontsize=14.5,
            color=COPPER, family=mono, weight="medium", va="center", zorder=5)
    ax.set_xlim(X0, X1); ax.set_ylim(y0, y1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("docs/assets/banner.png"))
    ap.add_argument("--snapshot", type=int, default=-1)
    ap.add_argument("--spacing", type=float, default=8.0, help="free-stream flux between streamlines, in cells")
    ap.add_argument("--vort-limit", type=float, default=6.0, help="colour range of vorticity, in U0 / chord")
    a = ap.parse_args()
    banner(a.flow, a.output, a.snapshot, a.spacing, a.vort_limit)
