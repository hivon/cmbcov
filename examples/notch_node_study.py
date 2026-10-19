#!/usr/bin/env python3
"""
Node plan of a ``map_filter.notch``, measured against exact rows.

For a mask symmetric about the scan pole, ``filtered_sum_rule_ratio(...,
method="alm")`` gives the exact white ratio ``g(l', d)`` of a node, with no
probe noise. This script computes it on a dense grid of ``l'`` through the
range a time-frequency notch can reach, then:

* the effect of the notch, ``g_notched / g_highpass - 1``;
* the interpolation error and the cost of candidate node plans, among them
  the plan ``filtering.default_nodes`` makes (graded about the onset
  ``m_lo / sin(theta_max)`` of the mask): PCHIP in ``ln l'`` (as
  ``rho_matrix``) through the exact values at the plan's nodes, against the
  exact values at every other reference row;
* the probe noise: random probes (``method="map"``) against the exact values
  at a few nodes.

Errors are reported for the separations the mask couples (``d`` up to its
coupling width and twice that): beyond them both sums are round-off and the
ratio is noise, multiplying covariance elements that are essentially zero.

The defaults are the case of docs/theory/filter_and_bin.md, "The notch":
``lmax`` 3500, sharp high-pass at 300, notch 4.03-4.09 Hz at 1 deg/s, a band
mask non-zero at colatitude 30-70 deg with 10 deg apodisations (full weight
40-60 deg, band-limit 256).
That takes about an hour on 8 cores; ``--quick`` runs a small version.

Usage
-----
    python examples/notch_node_study.py [--quick] [--f-lo 4.03 --f-hi 4.09
        --v-az 1.0 --lmax 3500 --lx 300] [--out notch_study.npz]
    python examples/notch_node_study.py --analyse notch_study.npz
"""

from __future__ import annotations

import argparse
import functools
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from cmbcov.assembly import filtering
from cmbcov.exact.common import mask_coupling_width
from cmbcov.harmonics.grid import gl_analysis, gl_shape, gl_thetas
from cmbcov.harmonics.sht import alm2cl

DEG = np.pi / 180


def _taper(x):
    x = np.clip(x, 0.0, 1.0)
    return np.sin(0.5 * np.pi * x) ** 2


#: Colatitudes (deg) where the band mask is non-zero; its apodisation width.
BAND, APOD = (30.0, 70.0), 10.0


def band_mask(lw, theta=BAND, apod=APOD):
    """alm of a band in colatitude (degrees), symmetric about the pole."""
    lg = 2 * lw
    th = gl_thetas(lg)
    w = _taper((th - theta[0] * DEG) / (apod * DEG))
    w = w * _taper((theta[1] * DEG - th) / (apod * DEG))
    return gl_analysis(np.repeat(w[:, None], gl_shape(lg)[1], axis=1), lw, lg)


def reference_rows(m_lo, lmax, base_nodes, quick):
    """Dense ``l'`` from ``0.95 m_lo`` to ``lmax - 1``: fine near the onset,
    coarser further up; plus the geometric nodes from ``0.75 m_lo``."""
    last = lmax - 1
    lo = int(0.95 * m_lo)
    if quick:
        rows = set(range(lo, last + 1, 2))
    else:
        rows = set(range(lo, lo + 160, 2)) | set(range(lo + 160, lo + 300, 5))
        rows |= set(range(lo + 300, lo + 600, 10)) | set(range(lo + 600, last, 25))
    rows |= {n for n in base_nodes if n >= 0.75 * m_lo} | {last}
    return sorted(n for n in rows if n <= last)


def compute(args):
    m_lo, m_hi = (
        filtering.scan_m(args.f_lo, args.v_az),
        filtering.scan_m(args.f_hi, args.v_az),
    )
    mask = band_mask(args.lw)
    width = mask_coupling_width(alm2cl(mask))
    highpass = functools.partial(filtering.highpass_profile, lx=args.lx, shape="sharp")
    notched = functools.partial(
        filtering.combined_profile,
        lx=args.lx,
        shape="sharp",
        notch={"m_lo": m_lo, "m_hi": m_hi},
    )
    base = filtering.node_plan(args.lmax, 32, 1.25, args.lx, nprobe_near_cut=128)
    rows = reference_rows(m_lo, args.lmax, base.nodes, args.quick)
    print(f"notch M {m_lo:.1f}-{m_hi:.1f}, coupling width {width}, {len(rows)} rows")

    def exact(profile, nodes):
        t0, out = time.time(), []
        for i, n in enumerate(nodes):
            g = filtering.filtered_sum_rule_ratio(
                mask, args.lw, profile, [n], args.dmax, method="alm"
            )["00"]
            out.append(g[0])
            print(
                f"  row {i + 1}/{len(nodes)} l'={n} {time.time() - t0:.0f} s",
                flush=True,
            )
        return np.array(out)

    hp_rows = rows[:: max(1, len(rows) // 25)]
    probe_rows = [rows[len(rows) // 20], rows[len(rows) // 3]]
    probes = {}
    for k in (32, 128):
        probes[f"g_map_{k}"] = filtering.filtered_sum_rule_ratio(
            mask, args.lw, notched, probe_rows, args.dmax, nprobe=k
        )["00"]
    np.savez(
        args.out,
        m=[m_lo, m_hi],
        lx=args.lx,
        dmax=args.dmax,
        sin_max=max(np.sin(t * DEG) for t in BAND) if BAND[1] < 90 else 1.0,
        width=width,
        lmax=args.lmax,
        base_nodes=base.nodes,
        rows=rows,
        g=exact(notched, rows),
        hp_rows=hp_rows,
        g_hp=exact(highpass, hp_rows),
        probe_rows=probe_rows,
        **probes,
    )
    return args.out


def analyse(path):
    r = np.load(path)
    m_lo, m_hi = r["m"]
    width, rows, last = int(r["width"]), r["rows"], int(r["lmax"]) - 1
    d_far = min(2 * width, r["g"].shape[1] - 1)
    row = {int(n): g for n, g in zip(rows, r["g"])}
    base = [int(n) for n in r["base_nodes"] if n in row]

    print(f"notch M {m_lo:.1f}-{m_hi:.1f}; {rows.size} exact rows {rows[0]}-{rows[-1]}")
    print("\neffect of the notch, g_notched / g_highpass - 1:")
    for n, g_hp in zip(r["hp_rows"], r["g_hp"]):
        e = row[int(n)] / g_hp - 1
        print(
            f"  l' {n:5d}: d=0 {e[0] * 100:+.3f}%   max d<={width} "
            f"{np.abs(e[: width + 1]).max() * 100:.3f}%"
        )

    def error(nodes, dmax):
        nodes = sorted(set(nodes))
        f = PchipInterpolator(np.log(nodes), np.array([row[n] for n in nodes]), axis=0)
        test = [n for n in rows if n not in nodes and n >= nodes[0]]
        diff = np.abs(f(np.log(test)) - np.array([row[n] for n in test]))
        return diff[:, : dmax + 1].max()

    def snap(targets):
        return sorted({int(rows[np.abs(rows - t).argmin()]) for t in targets})

    def geometric(ratio):
        t = 0.95 * m_lo * ratio ** np.arange(400)
        return base + snap(t[t <= last])

    def cost(nodes):
        """32 probes per node, ~ lmax_int^3 each, relative to ``base``."""
        lmax_int = int(r["dmax"]) + width

        def total(ns):
            return sum((n + lmax_int) ** 3 for n in set(ns))

        return total(nodes) / total(base)

    graded = filtering.default_nodes(
        last + 1,
        lx=float(r["lx"]),
        notch=(m_lo, m_hi),
        notch_sin_max=float(r["sin_max"]),
    )
    plans = {"geometric x1.25 only": base}
    for ratio in (1.10, 1.05, 1.03, 1.02):
        plans[f"x{ratio} from 0.95 m_lo"] = geometric(ratio)
    plans["default (graded)"] = base + snap([n for n in graded if n >= rows[0]])
    print(
        f"\nsin(theta_max) of the mask {float(r['sin_max']):.4f}: onset "
        f"{m_lo / float(r['sin_max']):.0f}; cost relative to the x1.25 nodes"
    )
    print(
        f"{'plan':26s} {'nodes':>5s} {'cost':>5s} {'d=0':>8s} {f'd<={width}':>8s} "
        f"{f'd<={d_far}':>8s}"
    )
    for name, nodes in plans.items():
        errs = [error(nodes, d) * 100 for d in (0, width, d_far)]
        print(
            f"{name:26s} {len(set(nodes)):5d} {cost(nodes):5.2f} "
            + " ".join(f"{e:7.3f}%" for e in errs)
        )

    print("\nprobe noise, |map - exact|:")
    for k in (32, 128):
        for j, n in enumerate(r["probe_rows"]):
            d = np.abs(r[f"g_map_{k}"][j] - row[int(n)])
            print(
                f"  {k:3d} probes, l'={n}: d=0 {d[0] * 100:.3f}%   "
                f"d<={width} {d[: width + 1].max() * 100:.3f}%"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--f-lo", type=float, default=4.03, help="Hz")
    p.add_argument("--f-hi", type=float, default=4.09, help="Hz")
    p.add_argument("--v-az", type=float, default=1.0, help="deg/s")
    p.add_argument("--lmax", type=int, default=3500)
    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--lw", type=int, default=256)
    p.add_argument("--dmax", type=int, default=99)
    p.add_argument(
        "--quick",
        action="store_true",
        help="lmax 400, lx 40, notch 0.5-0.52 Hz at 1 deg/s, lw 64",
    )
    p.add_argument("--out", default="notch_study.npz")
    p.add_argument("--analyse", metavar="NPZ", help="only analyse a saved study")
    args = p.parse_args()
    if args.analyse:
        analyse(args.analyse)
        return
    if args.quick:
        args.lmax, args.lx, args.f_lo, args.f_hi, args.lw, args.dmax = (
            400,
            40.0,
            0.5,
            0.52,
            64,
            30,
        )
    analyse(compute(args))


if __name__ == "__main__":
    main()
