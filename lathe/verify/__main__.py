"""
Demo and inspection entry point.

    python -m lathe.verify                    # every fixture, one report each
    python -m lathe.verify severed_crate      # one fixture
    python -m lathe.verify --json clean_prop  # machine form
    python -m lathe.verify --profile mobile_vr clean_prop
    python -m lathe.verify --table            # the rule table, all profiles

Exit code is the export gate: 0 when no error-severity finding fired.
"""

from __future__ import annotations

import argparse
import sys

from . import fixtures
from .engine import RULES, get_profile, load_profiles, load_rules, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lathe.verify")
    ap.add_argument("fixture", nargs="*", help="fixture name(s) from fixtures.EXPECTED")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the rule table and exit")
    ap.add_argument("--ascii", action="store_true", help="force ascii output")
    args = ap.parse_args(argv)

    if args.table:
        return _print_table()

    names = args.fixture or list(fixtures.EXPECTED)
    worst = 0
    for i, name in enumerate(names):
        builder = getattr(fixtures, name, None)
        if builder is None:
            print("no such fixture: %s" % name, file=sys.stderr)
            return 2
        report = run(builder(), args.profile)
        if args.json:
            print(report.to_json())
        else:
            if i:
                print("\n" + "=" * 78 + "\n")
            print("fixture %s   expected %s"
                  % (name, ", ".join(fixtures.EXPECTED.get(name, ())) or "no findings"))
            print(report.render(ascii_only=True if args.ascii else None))
        worst = max(worst, 0 if report.passed else 1)
    return worst


def _print_table() -> int:
    load_rules()
    load_profiles()
    from .engine import PROFILES

    print("%-6s %-6s %s" % ("code", "sev", "rule"))
    for code in sorted(RULES):
        spec = RULES[code]
        print("%-6s %-6s %s%s"
              % (code, spec.severity.label, spec.title,
                 "  [approximate]" if spec.is_approximate else ""))
    print()
    for name in sorted(PROFILES):
        prof = get_profile(name)
        print("%-18s %s" % (name, " ".join(prof.rules)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
