"""
Brief 12 Step 6 — render Fig 12.C overlay panels via ChimeraX CLI.

Two panels, picked by the orchestrator from the Brief-12 candidates list:
  (a) Short-H3 SUCCESS   — 7n9v_J  s0001 (H3 len 8,  scRMSD-H3 1.43 Å)
  (b) Long-H3  FAILURE   — 8elq_B  s0001 (H3 len 20, scRMSD-H3 10.94 Å)

Both from the expanded_pi_theta variant on the OLD test split.

Pipeline per panel:
  1. Open GT crystal PDB + model-generated PDB
  2. Show ONLY the heavy chain on both (hide antigen / light chains)
  3. matchmaker generated → GT on the heavy chain (sequence-based superposition)
  4. Colour GT in transparent grey; generated in slate; CDRs distinct
     (H1 salmon, H2 light-orange, H3 magenta — same palette as the
     existing Phase-B figures so the chapter feels cohesive)
  5. Camera focused on H3, lighting soft + silhouettes for cartoon clarity
  6. PNG at 1600×1200 with 3× supersampling

Inputs expected under data/eval/fig12c_inputs/ (rsync'd from Snellius):
  - 7n9v.pdb                            (GT crystal, chain J = heavy)
  - 8elq.pdb                            (GT crystal, chain B = heavy)
  - 7n9v_J_H3_sample_0001.pdb           (generated, chain J = heavy)
  - 8elq_B_H3_sample_0001.pdb           (generated, chain B = heavy)

ChimeraX binary is auto-detected from /Applications/ on macOS; override
with --chimerax. The per-panel .cxc files are written next to the PNG
output and NOT deleted, so the commands stay inspectable / reproducible.

Usage:
    python scripts/thesis/render_fig12c_overlays.py
    python scripts/thesis/render_fig12c_overlays.py \\
        --chimerax /Applications/ChimeraX-1.8.app/Contents/MacOS/ChimeraX
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS = PROJECT_ROOT / "data" / "eval" / "fig12c_inputs"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "figures" / "phase_b"

PANELS = [
    {
        "name": "fig12c_short_h3_success",
        "gt_pdb": INPUTS / "7n9v.pdb",
        "gen_pdb": INPUTS / "7n9v_J_H3_sample_0001.pdb",
        "gt_chain": "J",
        "gen_chain": "J",
        "h3_window": "95-102",
        "note": "7n9v_J s0001 — H3 len 8 (IVRYGY), scRMSD-H3 1.43 Å",
    },
    {
        "name": "fig12c_long_h3_failure",
        "gt_pdb": INPUTS / "8elq.pdb",
        "gen_pdb": INPUTS / "8elq_B_H3_sample_0001.pdb",
        "gt_chain": "B",
        "gen_chain": "B",
        "h3_window": "95-102",
        "note": "8elq_B s0001 — H3 len 20 (DASYDYLGYYYYYYADDY), "
                "scRMSD-H3 10.94 Å",
    },
]


def find_chimerax_mac() -> str | None:
    """Auto-detect ChimeraX on macOS via /Applications glob."""
    candidates = sorted(
        glob.glob("/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX"),
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_cxc(panel: dict, out_dir: Path, args) -> list[str]:
    out_png = out_dir / f"{panel['name']}.png"
    gt = panel["gt_chain"]
    gen = panel["gen_chain"]
    h3 = panel["h3_window"]
    # Paths may contain spaces (e.g. "/Users/.../Master Thesis/..."); ChimeraX
    # tokenises on whitespace so wrap every filename in literal double quotes.
    gt_pdb = panel["gt_pdb"]
    gen_pdb = panel["gen_pdb"]
    focus_sel = f"#2/{gen}" if args.focus == "chain" else f"#2/{gen}:{h3}"

    lines = [
        f"# {panel['note']}",
        f'open "{gt_pdb}"',
        f'open "{gen_pdb}"',
        # Restrict cartoons to the heavy chain on both models
        "hide cartoon",
        f"show #1/{gt} cartoon",
        f"show #2/{gen} cartoon",
        # Sequence-based superposition of generated onto GT, heavy chain only
        f"matchmaker #2/{gen} to #1/{gt}",
        # Colour scheme: GT light-grey w/ transparency, gen cornflower-blue,
        # CDRs distinct. CSS3-standard names (ChimeraX 1.12-rc rejects
        # `gray60` / `slate`).
        'color #1 "light gray"',
        "color #2 cornflowerblue",
        f"color #2/{gen}:26-32 salmon",
        f"color #2/{gen}:52-56 orange",
        f"color #2/{gen}:{h3} magenta",
        "hide atoms",
        "transparency #1 60 cartoons",
        # Background + presentation
        "set bgColor white",
        "graphics silhouettes true",
        "lighting soft",
        # Camera: focus, optional extra zoom, optional rotations
        f"view {focus_sel}",
    ]
    if args.zoom != 1.0:
        lines.append(f"zoom {args.zoom}")
    if args.turn_x:
        lines.append(f"turn x {args.turn_x}")
    if args.turn_y:
        lines.append(f"turn y {args.turn_y}")
    if args.turn_z:
        lines.append(f"turn z {args.turn_z}")
    lines.extend([
        f'save "{out_png}" width 1600 height 1200 supersample 3',
        "exit",
    ])
    return lines


def render_panel(panel: dict, chimerax_bin: str, out_dir: Path, args) -> None:
    cxc_lines = build_cxc(panel, out_dir, args)
    cxc_path = out_dir / f"{panel['name']}.cxc"
    cxc_path.write_text("\n".join(cxc_lines) + "\n")

    print(f"\n=== {panel['name']} ===")
    print(f"  {panel['note']}")
    print(f"  cxc: {cxc_path.relative_to(PROJECT_ROOT)}")
    # macOS-specific: ChimeraX 1.12-rc on Mac wasn't built with OSMesa, so
    # neither --nogui alone nor --nogui --offscreen creates a GL context for
    # the renderer ("Unable to save images because OpenGL rendering is not
    # available"). Workaround: drop --nogui so ChimeraX briefly opens a real
    # window (which gives it a NSOpenGL/Metal context). The .cxc ends with
    # `exit`, and --exit is belt-and-suspenders in case of a script error.
    # A window flashes for ~30-60s per render — no user interaction needed.
    cmd = [chimerax_bin, "--exit", str(cxc_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise SystemExit(
            f"FATAL: ChimeraX exited with code {result.returncode}"
        )
    out_png = out_dir / f"{panel['name']}.png"
    if not out_png.exists():
        print(f"  STDOUT (tail): {result.stdout[-400:]}")
        raise SystemExit(f"FATAL: expected PNG not found at {out_png}")
    print(f"  → {out_png.relative_to(PROJECT_ROOT)} "
          f"({out_png.stat().st_size // 1024} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render Fig 12.C overlays via ChimeraX CLI.",
    )
    ap.add_argument("--chimerax", default=None,
                    help="ChimeraX binary path. On macOS auto-detected from "
                         "/Applications/ChimeraX*.app.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help=f"Output dir for PNG + .cxc (default {DEFAULT_OUT})")
    ap.add_argument("--focus", choices=["chain", "h3"], default="chain",
                    help="Camera framing target. 'chain' (default) fits the "
                         "whole VHH heavy chain so the reader sees the "
                         "antibody silhouette with H3 highlighted in context. "
                         "'h3' frames just the H3 window (the OLD behaviour, "
                         "useful for a tight close-up).")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="Extra zoom applied AFTER the view-fit. <1 widens "
                         "(zoom out), >1 tightens (zoom in). Default 1.0.")
    ap.add_argument("--turn-x", type=float, default=0.0,
                    help="Camera rotation about X (degrees) — tilts up/down")
    ap.add_argument("--turn-y", type=float, default=0.0,
                    help="Camera rotation about Y (degrees) — pans left/right")
    ap.add_argument("--turn-z", type=float, default=0.0,
                    help="Camera rotation about Z (degrees) — rolls")
    ap.add_argument("--only", choices=["short", "long"], default=None,
                    help="Render only one panel (short = 7n9v_J success, "
                         "long = 8elq_B failure). Useful when iterating on "
                         "framing. Omit to render both.")
    args = ap.parse_args()

    chimerax = args.chimerax or find_chimerax_mac()
    if not chimerax or not Path(chimerax).exists():
        sys.exit(
            "FATAL: ChimeraX binary not found.\n"
            "  Pass --chimerax /path/to/ChimeraX, or install ChimeraX into "
            "/Applications/ChimeraX*.app/."
        )
    print(f"ChimeraX: {chimerax}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:  {out_dir}")

    panels_to_render = PANELS
    if args.only == "short":
        panels_to_render = [PANELS[0]]
    elif args.only == "long":
        panels_to_render = [PANELS[1]]

    for panel in panels_to_render:
        for k in ("gt_pdb", "gen_pdb"):
            if not panel[k].exists():
                sys.exit(
                    f"FATAL: input missing for panel '{panel['name']}':\n"
                    f"  {k} = {panel[k]}\n"
                    f"  Rsync from Snellius first; see brief 12 Step 6 "
                    f"instructions."
                )
        render_panel(panel, chimerax, out_dir, args)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
