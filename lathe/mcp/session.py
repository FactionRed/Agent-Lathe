"""
The session workspace -- DESIGN.md section 8, at the smallest size that works.

Every commit stores the *whole* program, not a delta. Operations are pure, so a
program is a complete description of its geometry and replaying it is the
definition of checking it out. Branching is therefore just alternative program
text, and comparing branch tips is comparing two programs' output.

Commits are content-addressed on (parent, program). Re-sending the program that
produced the current tip is a no-op, checked against the tip's text directly --
the hash alone cannot catch it, because after a commit the tip is its own parent
and the same program addresses differently.

A program is Python or `.lathe`. Python runs with the build API in scope; `.lathe`
is parsed and walked by lathe.lang, which is the only path where `600` is a type
error rather than a 600-metre crate. Both end in a Solid and a spec dict, so
everything downstream -- check, query, compare, export -- is language-blind.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import build as B
from ..build.bevel import collecting_reports
from ..build.solid import CardinalityError, LatheError, Solid


class ProgramError(LatheError):
    """The program failed. Carries the traceback line that matters."""


@dataclass(frozen=True)
class Commit:
    id: str
    parent: Optional[str]
    branch: str
    program: str
    message: str
    solid_name: str
    n_tris: int
    n_components: int
    volume: float
    geom_hash: str
    tags: Tuple[str, ...]
    spec: Dict[str, Any]
    created: float
    language: str = "python"
    # One note per bevel() whose delivered width fell short of the request. A
    # bevel that delivered in full is not worth a line, so absence means "fine".
    bevel_notes: Tuple[str, ...] = ()

    def short(self) -> str:
        return "%s  %-28s %6d tris  %d comp  %s" % (
            self.id, self.solid_name[:28], self.n_tris, self.n_components,
            self.message or "-")


LANGUAGES = ("auto", "python", "lathe")

# What a `.lathe` program's first statement looks like. Every alternative is two
# adjacent words (or a keyword and a string or brace), which is not valid Python,
# so a match is conclusive. A bare `uv = ...` or `mat = ...` is a Python
# assignment and does not match.
_LATHE_OPENER = re.compile(
    r'^(?:(?:param|let|unit|scene|mat|uv|joint|bind)\s+[A-Za-z_]\w*'
    r'|emit\s+"|solve\s*\{)')


def detect_language(program: str) -> str:
    """'lathe' if the first statement can only be .lathe, else 'python'."""
    for raw in program.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        return "lathe" if _LATHE_OPENER.match(line) else "python"
    return "python"


# `emit "NAME" { ... }` options that become the verifier's declared spec. Anything
# else is refused rather than dropped: an option that is accepted and ignored is
# a declaration the verifier never sees, which reads as a pass.
_EMIT_SPEC_KEYS = ("profile", "target", "budget", "origin", "components",
                   "texel_density", "texture_resolution")


def spec_from_emit(options: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(options) - set(_EMIT_SPEC_KEYS))
    if unknown:
        hint = ""
        if "lod" in unknown:
            hint = ("\n  `lod` is not accepted: nothing generates an LOD chain yet, "
                    "and an LOD index belongs in the emitted name (_LOD0).")
        raise ProgramError(
            "emit option(s) not understood: %s\n  Accepted: %s.%s"
            % (", ".join(unknown), ", ".join(_EMIT_SPEC_KEYS), hint))
    spec: Dict[str, Any] = {}
    for key, val in options.items():
        if key == "budget":
            if not isinstance(val, dict):
                raise ProgramError("emit budget must be a map, e.g. "
                                   "`budget: { tris_lod0: 1500 }`")
            spec[key] = {k: int(v) for k, v in val.items()}
        elif key in ("components", "texture_resolution"):
            spec[key] = int(val)
        elif key == "texel_density":
            spec[key] = float(val)
        else:
            spec[key] = str(val)
    return spec


def _api_namespace() -> Dict[str, Any]:
    """What a program can see.

    This is a *restricted namespace*, not a sandbox. `exec` runs with the real
    builtins, so a program that wants to reach the filesystem can. The threat
    model is the same as the caller already having a Python interpreter, which
    on a local stdio server it does. Do not expose this over a network.
    """
    ns: Dict[str, Any] = {
        "math": math,
        "np": np,
        "numpy": np,
    }
    for name in getattr(B, "__all__", []):
        ns[name] = getattr(B, name)
    # Imported here rather than at module scope: io depends on build, so the
    # arrow has to keep pointing one way.
    from ..io.glb import import_glb
    ns["import_glb"] = import_glb
    return ns


class Workspace:
    def __init__(self) -> None:
        self.commits: Dict[str, Commit] = {}
        self.branches: Dict[str, Optional[str]] = {"main": None}
        self.head: str = "main"
        self._solids: Dict[str, Solid] = {}
        self._order: List[str] = []

    # -- refs --------------------------------------------------------------

    def tip(self, branch: Optional[str] = None) -> Optional[str]:
        return self.branches[branch or self.head]

    def resolve(self, ref: Optional[str]) -> str:
        """Accepts a branch name, a commit id or prefix, HEAD, or HEAD~n."""
        if ref in (None, "", "HEAD"):
            cid = self.tip()
            if cid is None:
                raise LatheError(
                    "no commits yet on branch %r. Call lathe_apply first." % self.head)
            return cid
        if ref in self.branches:
            cid = self.branches[ref]
            if cid is None:
                raise LatheError("branch %r has no commits yet" % ref)
            return cid
        if ref.startswith("HEAD~"):
            n = int(ref[5:] or 1)
            cid = self.resolve("HEAD")
            for _ in range(n):
                parent = self.commits[cid].parent
                if parent is None:
                    raise LatheError(
                        "%s goes back further than history exists (%d commit(s))"
                        % (ref, len(self.commits)))
                cid = parent
            return cid
        hits = [c for c in self.commits if c.startswith(ref)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise LatheError(
                "unknown ref %r. Branches: %s. Commits: %s."
                % (ref, ", ".join(self.branches),
                   ", ".join(self._order[-6:]) or "none"))
        raise LatheError("ref %r is ambiguous: %s" % (ref, ", ".join(hits)))

    def solid(self, ref: Optional[str] = None) -> Solid:
        cid = self.resolve(ref)
        if cid not in self._solids:
            # Pure programs mean this is always reproducible.
            c = self.commits[cid]
            self._solids[cid] = self._run(c.program, c.language)[0]
        return self._solids[cid]

    def commit(self, ref: Optional[str] = None) -> Commit:
        return self.commits[self.resolve(ref)]

    # -- apply -------------------------------------------------------------

    def _run(self, program: str, language: str = "python"
             ) -> Tuple[Solid, Dict[str, Any], Tuple[str, ...]]:
        """Run a program. Returns (solid, spec, bevel notes).

        The notes are collected here, not read off the result Solid: the result
        is usually a boolean or a transform downstream of the bevel, and a report
        that only rode on the bevel's own Solid would be gone by then.
        """
        with collecting_reports() as reports:
            if language == "lathe":
                solid, spec = self._run_lathe(program)
            else:
                solid, spec = self._run_python(program)
        notes = tuple(
            r.describe() + "\n" + r.advice() for r in reports if r.clamped)
        return solid, spec, notes

    def _run_lathe(self, program: str) -> Tuple[Solid, Dict[str, Any]]:
        """Parse and walk a .lathe program. No exec, no builtins, no imports."""
        from ..lang import interp as L
        from ..lang.syntax import LatheSyntaxError
        from ..lang.values import UnitError
        try:
            solid, it = L.run(program)
        except LatheSyntaxError as exc:
            raise ProgramError(
                "syntax error in the .lathe program\n  %s" % exc) from None
        except (L.LatheRuntimeError, UnitError) as exc:
            raise ProgramError("%s" % exc) from None
        except LatheError:
            raise
        except Exception as exc:
            raise ProgramError(
                "%s while evaluating the .lathe program: %s"
                % (type(exc).__name__, exc)) from None
        return solid, spec_from_emit(it.emit_options)

    def _run_python(self, program: str) -> Tuple[Solid, Dict[str, Any]]:
        ns = _api_namespace()
        try:
            exec(compile(program, "<lathe program>", "exec"), ns)
        except B.CardinalityError:
            raise
        except LatheError:
            raise
        except Exception as exc:
            import traceback
            tb = traceback.extract_tb(exc.__traceback__)
            where = ""
            for frame in tb:
                if frame.filename == "<lathe program>":
                    where = "  line %d: %s\n" % (frame.lineno, (frame.line or "").strip())
            raise ProgramError(
                "%s: %s\n%s"
                "  The program runs with the lathe.build API in scope. Available: %s"
                % (type(exc).__name__, exc, where,
                   ", ".join(sorted(getattr(B, "__all__", [])))))
        if "result" not in ns:
            raise ProgramError(
                "the program did not assign `result`.\n"
                "  End it with `result = <Solid>`, e.g.\n"
                "    result = box(0.2, 0.2, 0.2, tag='body', name='SM_Test_LOD0')")
        solid = ns["result"]
        if not isinstance(solid, Solid):
            raise ProgramError(
                "`result` is %s, expected a Solid. Assign the solid itself, not "
                "a measurement of it." % type(solid).__name__)
        spec = ns.get("spec", {})
        if not isinstance(spec, dict):
            raise ProgramError("`spec` must be a dict, got %s" % type(spec).__name__)
        return solid, spec

    def apply(self, program: str, message: str = "",
              branch: Optional[str] = None,
              language: str = "auto") -> Tuple[Commit, Solid, str, bool]:
        """Run a program and commit its result. Returns (commit, solid, diff, is_new).

        `language` is "python", "lathe", or "auto", which reads the first
        statement (see detect_language). It is resolved before the branch is
        touched, so a bad value never leaves a half-made branch behind.
        """
        if language not in LANGUAGES:
            raise LatheError("language must be one of %s, got %r"
                             % (", ".join(LANGUAGES), language))
        lang = detect_language(program) if language == "auto" else language
        if branch is not None and branch not in self.branches:
            self.branches[branch] = self.tip()
            self.head = branch
        elif branch is not None:
            self.head = branch

        parent = self.tip()
        # Re-sending the program that produced the current tip changes nothing, so
        # it must not create a commit. Addressing on (parent, program) alone does
        # not catch this: once the program is applied, the tip *is* its own parent,
        # so the identical text hashes to a different address next time.
        if (parent is not None and self.commits[parent].program == program
                and self.commits[parent].language == lang):
            return self.commits[parent], self.solid(parent), "", False

        # Python ids keep their pre-.lathe addresses; only the new language is
        # salted, so identical text read as a different language is a new commit.
        salt = "" if lang == "python" else lang + "\x00"
        cid = hashlib.sha256(
            ((parent or "") + "\x00" + salt + program).encode("utf-8")).hexdigest()[:10]
        if cid in self.commits:
            return self.commits[cid], self.solid(cid), "", False

        solid, spec, notes = self._run(program, lang)
        msh = solid.mesh()
        geom = hashlib.sha256(
            msh.verts.tobytes() + msh.tris.tobytes()).hexdigest()[:12]

        c = Commit(
            id=cid, parent=parent, branch=self.head, program=program,
            message=message, solid_name=solid.name, n_tris=solid.n_tris,
            n_components=solid.components(), volume=solid.volume(),
            geom_hash=geom, tags=tuple(sorted(set(solid.tags.values()))),
            spec=dict(spec), created=time.time(), language=lang,
            bevel_notes=notes,
        )
        self.commits[cid] = c
        self._solids[cid] = solid
        self.branches[self.head] = cid
        self._order.append(cid)

        diff = ""
        if parent is not None:
            diff = "\n".join(difflib.unified_diff(
                self.commits[parent].program.splitlines(),
                program.splitlines(),
                fromfile=parent, tofile=cid, lineterm="", n=2))
        return c, solid, diff, True

    # -- branching ---------------------------------------------------------

    def new_branch(self, name: str, from_ref: Optional[str] = None) -> str:
        if name in self.branches:
            raise LatheError(
                "branch %r already exists at %s. Use lathe_checkout to switch."
                % (name, self.branches[name]))
        self.branches[name] = self.resolve(from_ref) if (
            from_ref or self.tip()) else None
        self.head = name
        return name

    def checkout(self, ref: str) -> str:
        if ref in self.branches:
            self.head = ref
            return ref
        cid = self.resolve(ref)
        # Detached inspection is not worth modelling: name it so it is a branch.
        name = "at-%s" % cid
        self.branches[name] = cid
        self.head = name
        return name

    def log(self, limit: int = 20, branch: Optional[str] = None) -> List[Commit]:
        cid = self.branches.get(branch or self.head)
        out: List[Commit] = []
        while cid is not None and len(out) < limit:
            c = self.commits[cid]
            out.append(c)
            cid = c.parent
        return out

    def geometry_changed(self, a: str, b: str) -> bool:
        return self.commit(a).geom_hash != self.commit(b).geom_hash
