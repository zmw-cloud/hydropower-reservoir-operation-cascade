"""Render only the previously requested paper figures into fresh output folders."""
from datetime import datetime
from pathlib import Path
import argparse
import os
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

def native(p):
    p = Path(p).absolute()
    return Path("\\\\?\\" + str(p)) if os.name == "nt" and not str(p).startswith("\\\\?\\") else p

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--figures", nargs="+", choices=("all", "hydrology", "tradeoff", "violin", "power", "levels", "spill", "adjustment", "transfer"))
    p.add_argument("--font-dir", type=Path)
    args = p.parse_args()
    from audit_repository import audit
    result = audit(plots=True)
    print(f"FROZEN_INPUT_CHECKS={result['status']}")
    if args.check_only:
        return
    if not args.figures:
        p.error("Choose --figures explicitly, or use --check-only")
    selected = set(args.figures)
    if "all" in selected:
        selected = {"hydrology", "tradeoff", "violin", "power", "levels", "spill", "adjustment", "transfer"}
    out = ROOT / "outputs/figures" / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    native(out).mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    font = ["--font-dir", str(args.font_dir.absolute())] if args.font_dir else []
    ids = {"hydrology": "fig1", "tradeoff": "fig3", "power": "fig5", "levels": "fig6", "spill": "fig7", "adjustment": "fig8"}
    chosen = [ids[k] for k in ids if k in selected]
    calls = []
    if chosen:
        calls.append(["render_anonymized_paper_figures.py", "--figures", *chosen, "--output-dir", str(out / "main"), *font])
    new = selected & {"violin", "transfer"}
    if new:
        figure = "both" if len(new) == 2 else next(iter(new))
        calls.append(["plot_paper_results.py", "--figure", figure, "--output-dir", str(out / "annual_and_transfer"), *font])
    for cmd in calls:
        subprocess.run([sys.executable, "-B", str(native(ROOT / "scripts" / cmd[0])), *cmd[1:]],
                       cwd=native(ROOT), env=env, check=True)
    print("FIGURE_OUTPUT=" + str(out))

if __name__ == "__main__":
    main()
