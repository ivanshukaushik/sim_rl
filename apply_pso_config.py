#!/usr/bin/env python3
"""
apply_pso_config.py — Write PSO-optimal parameters into config.py.

Usage:
    python apply_pso_config.py                        # uses pso_results/best_config.json
    python apply_pso_config.py path/to/best_config.json
"""

import json
import re
import sys
import os


def apply(json_path: str, config_path: str = "config.py") -> None:
    with open(json_path) as f:
        best = json.load(f)

    with open(config_path) as f:
        src = f.read()

    updated = dict(best)
    changed = []

    for name, new_val in updated.items():
        # Match:  name: <type> = <old_value>  # optional comment
        pattern = rf"(    {re.escape(name)}\s*:\s*\w+\s*=\s*)([^\s#\n]+)"
        if isinstance(new_val, float):
            replacement = rf"\g<1>{new_val:.6g}"
        else:
            replacement = rf"\g<1>{new_val}"
        new_src, n = re.subn(pattern, replacement, src)
        if n > 0:
            # Find old value for reporting
            m = re.search(pattern, src)
            old_val = m.group(2) if m else "?"
            changed.append((name, old_val, new_val))
            src = new_src
        else:
            print(f"  [WARN] Could not find '{name}' in {config_path} — skipped")

    with open(config_path, "w") as f:
        f.write(src)

    print(f"Applied {len(changed)} parameter(s) from {json_path} → {config_path}")
    print()
    print(f"  {'Parameter':<30s}  {'Old':>12s}  →  {'New':>12s}")
    print(f"  {'-'*30}  {'-'*12}     {'-'*12}")
    for name, old, new in changed:
        print(f"  {name:<30s}  {str(old):>12s}  →  {str(new):>12s}")


def main() -> None:
    json_path = sys.argv[1] if len(sys.argv) > 1 else "pso_results/best_config.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        print("Run pso_optimize.py first, or pass the path to a best_config.json.")
        sys.exit(1)
    apply(json_path)


if __name__ == "__main__":
    main()
