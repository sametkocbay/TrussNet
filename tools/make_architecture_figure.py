#!/usr/bin/env python
"""Generate the architecture overview used by README.md and docs/METHOD.md, in a light and a dark variant.

    python tools/make_architecture_figure.py
    python tools/make_architecture_figure.py --theme dark --preview /tmp/architecture.png
    python tools/make_architecture_figure.py --output-dir /tmp/architecture-review

Uses matplotlib, like tools/make_improvement_figure.py; no model, GPU or data required.
Writes architecture.svg (light) and architecture_dark.svg with transparent backgrounds, so each sits directly on
GitHub's light or dark page. Rendered text bounds are checked before writing. SVG glyphs are embedded as paths so
the type sizes and line breaks remain consistent across renderers. The title uses JetBrains Mono when it is
installed or found in TRUSSNET_FONT_DIR, and DejaVu Sans Mono otherwise.
"""
from __future__ import annotations

import argparse
import os
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / 'docs' / 'figures'
FONT_DIR = Path(os.environ.get('TRUSSNET_FONT_DIR', Path.home() / '.local/share/fonts/JetBrainsMono-2.304/fonts/ttf'))
WIDTH, HEIGHT = 960, 1040   # main column at x = 480, the horizontal center
# Two variants with a transparent background, so the figure sits directly on GitHub's light or dark page.
THEMES = {
    'light': dict(ink='#1d2733', muted='#4a5a6e', edge='#9aabbd', rule='#c8d3df', accent={'core': '#b5651d', 'decode': '#6a4c93'},
                  fill={'input': '#eef4fb', 'physics': '#eff7ef', 'core': '#fdf5ec', 'decode': '#f4effa', 'white': '#ffffff'}),
    'dark': dict(ink='#eef2f6', muted='#8fa1b3', edge='#3a4d61', rule='#3a4d61', accent={'core': '#d9824b', 'decode': '#a58fd8'},
                 fill={'input': '#122435', 'physics': '#12282a', 'core': '#251c16', 'decode': '#1d1a2e', 'white': '#0f1b27'}),
}
INK = MUTED = EDGE = RULE = PALETTE = ACCENT = None     # set by set_theme()


def set_theme(name):
    global INK, MUTED, EDGE, RULE, PALETTE, ACCENT
    t = THEMES[name]
    INK, MUTED, EDGE, RULE = t['ink'], t['muted'], t['edge'], t['rule']
    PALETTE, ACCENT = t['fill'], t['accent']


def mono_family():
    """JetBrains Mono for the title if available, else the monospace font that ships with matplotlib."""
    path = FONT_DIR / 'JetBrainsMono-ExtraBold.ttf'
    if path.is_file():
        font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    return 'JetBrains Mono' if 'JetBrains Mono' in names else 'DejaVu Sans Mono'


def architecture():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'path',
                         'svg.hashsalt': 'trussnet-architecture', 'savefig.transparent': True})
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    checks = []

    # Layout coordinates and type sizes use the same pixel scale. Matplotlib
    # takes point sizes, so convert explicitly rather than mixing both units.
    def text(value, x, y, pixels=15, bold=False, color=INK, ha='center'):
        return ax.text(x, y, value, ha=ha, va='center',
                       fontsize=pixels * 72 / 100, fontweight='bold' if bold else 'normal',
                       color=color, linespacing=1.45)

    def box(x, y, w, h, theme='white', accent=False):
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0,rounding_size=10', linewidth=1.4,
            edgecolor=ACCENT.get(theme, MUTED) if accent else EDGE,
            facecolor=PALETTE[theme])
        ax.add_patch(patch)
        return patch

    def label(patch, value, x, y, pixels=15, bold=False, color=INK, ha='center'):
        item = text(value, x, y, pixels, bold, color, ha)
        checks.append((patch, item))
        return item

    def arrow(x1, y1, x2, y2, color=MUTED, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
            mutation_scale=13, linewidth=1.5, color=color,
            linestyle=(0, (5, 4)) if dashed else '-', shrinkA=0, shrinkB=0))

    center, left, width = 480, 210, 540
    text('TrussNet', center, 28, 23, True).set_fontfamily(mono_family())
    text('One member shown · 3.44 M parameters per member', center, 55, 12.5, color=MUTED)

    patch = box(250, 82, 460, 63, 'input')
    label(patch, 'Input · 20 past velocity frames', center, 103, 15, True)
    label(patch, 'u and v on a 32 × 64 grid', center, 127, 12.5, color=MUTED)
    arrow(center, 145, center, 175)

    patch = box(left, 175, width, 75, 'physics')
    label(patch, 'Physics preprocessing', center, 198, 15, True)
    label(patch, 'Freestream normalization · mask · missing-vector maps', center, 226, 12.5, color=MUTED)
    arrow(center, 250, center, 280)

    patch = box(left, 280, width, 75, 'physics')
    label(patch, 'Baseline forecast', center, 303, 15, True)
    label(patch, 'Blend the latest field with a downstream-motion forecast', center, 329, 12.5, color=MUTED)
    arrow(center, 355, center, 385)

    core = box(190, 385, 580, 255, 'core', accent=True)
    label(core, 'Axial transformer', center, 405, 15, True, ACCENT['core'])
    patch = box(225, 423, 510, 49)
    label(patch, 'Fold past frames into channels; form 1 × 2 spatial patches', center, 440, 12.5, color=MUTED)
    label(patch, '1,024 tokens × 128 features', center, 457, 12.5, True)
    arrow(center, 472, center, 493)
    patch = box(225, 493, 510, 61)
    label(patch, '6 blocks · x attention → y attention → MLP', center, 512, 13, True)
    label(patch, '8 attention heads · FiLM conditioning', center, 537, 12.5, color=MUTED)
    arrow(center, 554, center, 576)
    patch = box(225, 576, 510, 46)
    label(patch, 'Restore features for every future frame and cell', center, 599, 12.5, color=MUTED)

    # The context route stays outside every main-path box. Its final arrow
    # terminates at the block it conditions, not at the enclosing container.
    context = box(25, 483, 140, 82, 'core')
    label(context, 'Flow context', 95, 501, 13, True)
    label(context, '8 numbers → MLP', 95, 525, 12, color=MUTED)
    label(context, 'shared across blocks', 95, 548, 11.5, color=MUTED)
    ax.plot([210, 95, 95], [212, 212, 462], color=ACCENT['core'], linewidth=1.5,
            linestyle=(0, (5, 4)), solid_capstyle='butt')
    arrow(95, 462, 95, 483, ACCENT['core'], dashed=True)
    arrow(165, 524, 225, 524, ACCENT['core'], dashed=True)
    arrow(center, 640, center, 670)

    heads = box(190, 670, 580, 107, 'input')
    label(heads, 'Output heads', center, 690, 15, True)
    for x, title, line1, line2 in (
        (290, 'Central field', 'Baseline +', 'learned correction'),
        (480, 'Fluctuation', 'Zero average over', 'the 20 future frames'),
        (670, 'Quantiles', '5 ordered percentiles', 'for each u and v value'),
    ):
        label(heads, title, x, 720, 13, True)
        label(heads, line1, x, 741, 12, color=MUTED)
        label(heads, line2, x, 759, 12, color=MUTED)
    # Thin dividers give equal space to the three outputs without implying
    # three separate learned layers: the code uses one shared output layer.
    for x in (385, 575):
        ax.plot([x, x], [711, 765], color=RULE, linewidth=0.9)
    arrow(center, 777, center, 807)

    ensemble = box(190, 807, 580, 141, 'decode', accent=True)
    label(ensemble, 'Two-member ensemble decode', center, 828, 15, True, ACCENT['decode'])
    for x, title, lines in (
        (335, 'Velocity forecast', ('Average central fields', 'Rescale averaged fluctuations', 'Apply forecast adjustments')),
        (625, 'Uncertainty bounds', ('Average corresponding quantiles', 'Select interval width', 'Center on predicted median')),
    ):
        label(ensemble, title, x, 859, 13, True)
        for i, line in enumerate(lines):
            label(ensemble, f'• {line}', x, 888 + 18 * i, 12, color=MUTED)
    ax.plot([480, 480], [850, 936], color=RULE, linewidth=0.9)
    arrow(center, 948, center, 978)

    patch = box(250, 978, 460, 51, 'input')
    label(patch, '20 future velocity frames + lower and upper bounds', center, 995, 13, True)
    label(patch, 'Physical units · 32 × 64 grid', center, 1015, 12, color=MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, item in checks:
        bounds, bb = patch.get_window_extent(renderer), item.get_window_extent(renderer)
        if not (bb.x0 >= bounds.x0 + 8 and bb.x1 <= bounds.x1 - 8 and
                bb.y0 >= bounds.y0 + 7 and bb.y1 <= bounds.y1 - 7):
            raise RuntimeError(f'Text does not fit its box: {item.get_text()!r}')
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=FIGURES, help='Output directory (default: docs/figures).')
    ap.add_argument('--theme', choices=['light', 'dark', 'both'], default='both')
    ap.add_argument('--preview', type=Path, help='Optional PNG render for visual inspection.')
    args = ap.parse_args()
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    for name in themes:
        set_theme(name)
        fig = architecture()
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', metadata={'Date': None, 'Title': 'TrussNet architecture',
            'Description': 'Physics scaffold, axial transformer, output heads and two-member ensemble decode.'})
        data = ('\n'.join(line.rstrip() for line in buf.getvalue().decode('utf-8').splitlines()) + '\n').encode('utf-8')
        path = args.output_dir / ('architecture.svg' if name == 'light' else 'architecture_dark.svg')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(path)
        if args.preview:
            preview = args.preview if len(themes) == 1 else args.preview.with_name(f'{args.preview.stem}_{name}{args.preview.suffix}')
            preview.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview, dpi=130, transparent=False, facecolor='#ffffff' if name == 'light' else '#0d1117')
            print(preview)
        plt.close(fig)


if __name__ == '__main__':
    main()
