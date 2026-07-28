"""
Runner for the bevel/fillet spike.

    python run_all.py                    # run every candidate that is available
    python run_all.py --only C           # one candidate
    python run_all.py --json out.json    # keep the raw numbers

Each candidate runs as a subprocess under whatever interpreter it needs, prints
its own readable report, and emits a JSON blob after a @@RESULT@@ marker. That
is the only way to run them uniformly: approach A needs the OCCT venv and
approach B needs Blender's own Python for its inner half.

Interpreter discovery, in order:
  A  $LATHE_OCCT_PYTHON, then this interpreter if OCP imports, then the venv
     paths in OCCT_CANDIDATES below
  B  this interpreter (it shells out to $LATHE_BLENDER itself)
  C  this interpreter
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OCCT_CANDIDATES = [
    os.environ.get("LATHE_OCCT_PYTHON"),
    os.path.join(HERE, "occtvenv", "Scripts", "python.exe"),
    os.path.join(HERE, "..", "..", "occtvenv", "Scripts", "python.exe"),
    r"C:\Users\sydne\AppData\Local\Temp\claude"
    r"\C--Users-sydne-ClaudCode-Agent-3D"
    r"\0ba6fe10-56a7-4525-a8b4-a19f1f5a9fbe\scratchpad\occtvenv"
    r"\Scripts\python.exe",
]


def can_import(python, module):
    if not python or not os.path.exists(python):
        return False
    p = subprocess.run([python, "-c", "import %s" % module],
                       capture_output=True, text=True)
    return p.returncode == 0


def find_occt_python():
    if can_import(sys.executable, "OCP"):
        return sys.executable
    for cand in OCCT_CANDIDATES:
        if cand and can_import(cand, "OCP") and can_import(cand, "manifold3d"):
            return cand
    return None


def run_script(python, script, label):
    print("\n>>> %s   (%s)" % (label, python))
    p = subprocess.run([python, os.path.join(HERE, script)],
                       capture_output=True, text=True)
    out = p.stdout
    payload = None
    if "@@RESULT@@" in out:
        head, _, tail = out.partition("@@RESULT@@")
        print(head.rstrip())
        try:
            payload = json.loads(tail.strip())
        except json.JSONDecodeError as exc:
            print("  could not parse result JSON: %s" % exc)
    else:
        print(out.rstrip())
    if p.returncode != 0:
        print("  EXIT %d\n%s" % (p.returncode, p.stderr[-4000:]))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "C"], action="append")
    ap.add_argument("--json")
    args = ap.parse_args()
    want = set(args.only or ["A", "B", "C"])
    results = {}

    C.rule("BEVEL / FILLET SPIKE -- fixture and referee")
    fx = C.Fixture()
    man, bid, cid = fx.manifold()
    v, t, g = C.manifold_arrays(man, bid, cid)
    chk = C.check_mesh(v, t, "fixture")
    C.kv("fixture", repr(fx))
    C.kv("triangles / logical faces",
         "%d / %d" % (chk["tris"], C.coplanar_groups(v, t)[1]))
    C.kv("components / genus / volume",
         "%d / %s / %.6f" % (chk["components"], chk["genus"], chk["volume"]))
    C.kv("faces(#cut_face) baseline",
         C.before_snapshot(fx, v, t, g)["sel_cut_any_faces"])
    C.kv("referee ground truth", C.referee(fx, v, t)["counts"])

    if "A" in want:
        py = find_occt_python()
        if py is None:
            C.rule("APPROACH A -- SKIPPED")
            print("  No interpreter with both OCP and manifold3d was found.\n"
                  "  Create one with:\n"
                  "    py -3.12 -m venv occtvenv\n"
                  "    occtvenv\\Scripts\\python -m pip install cadquery-ocp "
                  "manifold3d\n"
                  "  then set LATHE_OCCT_PYTHON to it, or place it at "
                  "prototype/bevel_spike/occtvenv.")
            results["A"] = {"skipped": "no OCP interpreter"}
        else:
            results["A"] = run_script(py, "approach_a_occt.py",
                                      "APPROACH A -- OpenCASCADE")

    if "B" in want:
        results["B"] = run_script(sys.executable, "approach_b_bmesh.py",
                                  "APPROACH B -- bmesh")

    if "C" in want:
        results["C"] = run_script(sys.executable, "approach_c_sdf.py",
                                  "APPROACH C -- SDF / implicit")

    summarise(results)
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1, default=str)
        print("\nraw results written to %s" % args.json)


def summarise(results):
    C.rule("HEADLINE -- the drift test from DESIGN.md section 13")
    print("  faces(#cut_face) must stay at 3 for the slotted-cube fixture.\n")
    rows = []
    a = results.get("A") or {}
    if a and "A1_provenance" in a:
        p = a["A1_provenance"]
        rows.append(("A  OCCT fillet, B-Rep history",
                     p["before"]["sel_cut_any_faces"], p["sel_cut_any_faces"],
                     p["false_positive_tris"], a["A1_check"]["tris"],
                     "%.2f s" % a["A1_total_s"]))
    b = results.get("B") or {}
    for mode in ("raw", "inherit", "exclusive"):
        e = (b.get("B_tag_modes") or {}).get(mode)
        if e and "provenance" in e:
            p = e["provenance"]
            rows.append(("B  bmesh bevel, tag_mode=%s" % mode,
                         p["before"]["sel_cut_any_faces"],
                         p["sel_cut_any_faces"],
                         p["false_positive_real_tris"], e["check"]["tris"],
                         "%.2f s" % e["blender_info"]["wall_s"]))
    c = results.get("C") or {}
    for k, e in (c.get("C1_level_set") or {}).items():
        if "error" in e or "rounded" not in k:
            continue
        rows.append(("C  level_set %s" % k,
                     e["sel_cut_faces_before"],
                     e["sel_cut_faces_after_reattribution"],
                     "n/a", e["tris"], "%.2f s" % e["level_set"]["seconds"]))
    for k, e in (c.get("C3_minkowski") or {}).items():
        if "error" in e or "16" not in k:
            continue
        rows.append(("C  minkowski %s" % k, 3,
                     e["sel_cut_faces_after_reattribution"],
                     e.get("false_positive_real_tris", "n/a"), e["tris"],
                     "%.2f s" % e["minkowski"]["seconds"]))

    print("  %-34s %6s %6s %8s %8s %9s"
          % ("approach", "before", "after", "wrong", "tris", "time"))
    print("  " + "-" * 76)
    for name, before, after, fp, tris, t in rows:
        flag = "  <-- DRIFT" if before != after else ""
        print("  %-34s %6s %6s %8s %8s %9s%s"
              % (name, before, after, fp, tris, t, flag))


if __name__ == "__main__":
    main()
