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

A workspace is shared by every tool call, and lathe_apply runs programs on worker
threads, so state changes under a lock and *only on success*: a program that fails
leaves no branch, no moved head and no commit, and two programs running at once
each land on the branch they were sent to. (Before that, a slow call on one branch
overlapping a fast call on another lost both commits.)
"""

from __future__ import annotations

import difflib
import hashlib
import math
import re
import threading
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


_BRANCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$")


def check_branch_name(name: str) -> str:
    """Branch names are short, printable and cannot be read as a ref expression."""
    if not isinstance(name, str) or not _BRANCH_NAME.match(name) or name == "HEAD":
        raise LatheError(
            "%r is not a usable branch name. Use 1-64 characters from letters, "
            "digits, '.', '_', '-' and '/', starting with a letter or digit, and "
            "not 'HEAD'." % (name,))
    return name


def _require_body(solid: Solid) -> None:
    """A solid with no triangles or no volume is not an asset.

    box(0mm, ...) and box(-5mm, ...) produced an empty Manifold that committed,
    and then every later tool failed with "max() iterable argument is empty".
    """
    if solid.n_tris == 0 or not solid.volume() > 0.0:
        raise ProgramError(
            "the program's result is empty (%d triangles, volume %g). Zero or "
            "negative dimensions, or a boolean that removed everything, are the "
            "usual causes." % (solid.n_tris, solid.volume()))


class ApplyResult(tuple):
    """(commit, solid, diff, is_new), unpackable as that, plus where it landed.

    `branch` is the branch the call committed on, which is not `commit.branch`
    when the program already existed as a commit made somewhere else. `moved` is
    true when no commit was made but the branch tip moved onto an existing one.
    """

    def __new__(cls, commit, solid, diff, is_new, branch, moved=False):
        self = super().__new__(cls, (commit, solid, diff, is_new))
        self.branch = branch
        self.moved = moved
        return self


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
        # Re-entrant: public methods call each other. Never held while a program
        # runs, which takes seconds.
        self._lock = threading.RLock()

    # -- refs --------------------------------------------------------------

    def tip(self, branch: Optional[str] = None) -> Optional[str]:
        with self._lock:
            name = branch or self.head
            if name not in self.branches:
                raise LatheError("unknown branch %r. Branches: %s."
                                 % (name, ", ".join(sorted(self.branches))))
            return self.branches[name]

    def resolve(self, ref: Optional[str]) -> str:
        with self._lock:
            return self._resolve(ref)

    def _resolve(self, ref: Optional[str]) -> str:
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
            # HEAD~, HEAD~3, HEAD~1~1. Anything else (HEAD~abc, HEAD~-1) used to
            # reach int() and either raise a bare ValueError or, for a negative,
            # quietly mean HEAD.
            if not re.fullmatch(r"HEAD(?:~\d*)+", ref):
                raise LatheError(
                    "%r is not a ref. HEAD~n goes back n commits (HEAD~, HEAD~2, "
                    "HEAD~1~1); n is a whole number." % ref)
            n = sum(int(t or 1) for t in re.findall(r"~(\d*)", ref))
            cid = self._resolve("HEAD")
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
        with self._lock:
            cid = self.resolve(ref)
            hit = self._solids.get(cid)
            c = self.commits[cid]
        if hit is not None:
            return hit
        # Pure programs mean this is always reproducible. Run outside the lock.
        built = self._run(c.program, c.language)[0]
        with self._lock:
            return self._solids.setdefault(cid, built)

    def commit(self, ref: Optional[str] = None) -> Commit:
        with self._lock:
            return self.commits[self.resolve(ref)]

    # -- apply -------------------------------------------------------------

    def _run(self, program: str, language: str = "python"
             ) -> Tuple[Solid, Dict[str, Any], Tuple[str, ...]]:
        """Run a program. Returns (solid, spec, bevel notes).

        The notes are collected here, not read off the result Solid: the result
        is usually a boolean or a transform downstream of the bevel, and a report
        that only rode on the bevel's own Solid would be gone by then.
        """
        lang_notes: Tuple[str, ...] = ()
        with collecting_reports() as reports:
            if language == "lathe":
                solid, spec, lang_notes = self._run_lathe(program)
            else:
                solid, spec = self._run_python(program)
        notes = lang_notes + tuple(
            r.describe() + "\n" + r.advice() for r in reports if not r.ok)
        return solid, spec, notes

    def _run_lathe(self, program: str
                   ) -> Tuple[Solid, Dict[str, Any], Tuple[str, ...]]:
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
        _require_body(solid)
        notes = tuple("%s\n%s" % (head, advice) for head, advice in it.notes)
        return solid, spec_from_emit(it.emit_options), notes

    def _run_python(self, program: str) -> Tuple[Solid, Dict[str, Any]]:
        ns = _api_namespace()
        try:
            exec(compile(program, "<lathe program>", "exec"), ns)
        except B.CardinalityError:
            raise
        except LatheError:
            raise
        except SystemExit as exc:
            # exit(), quit() and sys.exit() raise this, and `except Exception`
            # does not catch it: the MCP server died and the workspace with it.
            raise ProgramError(
                "the program called exit()/sys.exit(%s). A program builds a "
                "solid and assigns `result`; it does not end the session."
                % (exc.code if exc.code is not None else "")) from None
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
        _require_body(solid)
        return solid, spec

    def apply(self, program: str, message: str = "",
              branch: Optional[str] = None,
              language: str = "auto") -> "ApplyResult":
        """Run a program and commit its result.

        Returns an ApplyResult, which unpacks as (commit, solid, diff, is_new).

        Nothing changes unless the program succeeds. A new branch is created, and
        head moved, only at the point of committing; a program that fails leaves
        the workspace exactly as it found it. The branch is the one named, or the
        head *at the time of the call*, so concurrent calls land where they were
        sent.
        """
        if language not in LANGUAGES:
            raise LatheError("language must be one of %s, got %r"
                             % (", ".join(LANGUAGES), language))
        if branch is not None:
            check_branch_name(branch)
        lang = detect_language(program) if language == "auto" else language
        salt = "" if lang == "python" else lang + "\x00"

        with self._lock:
            target = branch if branch is not None else self.head
            # A new branch forks from the head's tip as of now.
            fork = (self.branches[target] if target in self.branches
                    else self.branches[self.head])
            # Re-sending the program that produced the tip changes nothing, so it
            # must not create a commit. Addressing on (parent, program) alone does
            # not catch this: once the program is applied, the tip *is* its own
            # parent, so the identical text hashes to a different address.
            if (fork is not None and self.commits[fork].program == program
                    and self.commits[fork].language == lang):
                self._land(target, fork)
                return ApplyResult(self.commits[fork], self._solid_of(fork), "",
                                   False, target)

        solid, spec, notes = self._run(program, lang)
        msh = solid.mesh()
        geom = hashlib.sha256(
            msh.verts.tobytes() + msh.tris.tobytes()).hexdigest()[:12]
        n_components, volume = solid.components(), solid.volume()
        tags = tuple(sorted(set(solid.tags.values())))

        with self._lock:
            # Another call may have moved this branch while the program ran. Commit
            # on top of wherever it is now: calls serialise, none is lost.
            parent = self.branches[target] if target in self.branches else fork
            cid = hashlib.sha256(
                ((parent or "") + "\x00" + salt + program).encode("utf-8")
            ).hexdigest()[:10]
            if cid in self.commits:
                # The same program on the same parent already exists, made on some
                # other branch. This branch still has to move onto it: returning
                # the existing commit and leaving the tip put left the branch where
                # it was and told the caller nothing had changed.
                moved = self.branches.get(target) != cid
                self._land(target, cid)
                return ApplyResult(self.commits[cid], self._solid_of(cid), "",
                                   False, target, moved=moved)
            c = Commit(
                id=cid, parent=parent, branch=target, program=program,
                message=message, solid_name=solid.name, n_tris=solid.n_tris,
                n_components=n_components, volume=volume, geom_hash=geom,
                tags=tags, spec=dict(spec), created=time.time(), language=lang,
                bevel_notes=notes,
            )
            self.commits[cid] = c
            self._solids[cid] = solid
            self._order.append(cid)
            self._land(target, cid)
            diff = ""
            if parent is not None:
                diff = "\n".join(difflib.unified_diff(
                    self.commits[parent].program.splitlines(),
                    program.splitlines(),
                    fromfile=parent, tofile=cid, lineterm="", n=2))
            return ApplyResult(c, solid, diff, True, target)

    def _land(self, branch: str, cid: Optional[str]) -> None:
        """Point `branch` at `cid`, creating it, and make it the head."""
        with self._lock:
            self.branches[branch] = cid
            self.head = branch

    def _solid_of(self, cid: str) -> Solid:
        """The Solid of a commit that exists, without running anything twice."""
        got = self._solids.get(cid)
        if got is not None:
            return got
        return self.solid(cid)

    # -- branching ---------------------------------------------------------

    def new_branch(self, name: str, from_ref: Optional[str] = None) -> str:
        check_branch_name(name)
        with self._lock:
            if name in self.branches:
                raise LatheError(
                    "branch %r already exists at %s. Use lathe_checkout to switch."
                    % (name, self.branches[name]))
            self.branches[name] = self._resolve(from_ref) if (
                from_ref or self.tip()) else None
            self.head = name
            return name

    def checkout(self, ref: str) -> str:
        with self._lock:
            if ref in self.branches:
                self.head = ref
                return ref
            cid = self._resolve(ref)
            # Detached inspection is not worth modelling: name it so it is a
            # branch. Going back to a commit you already checked out switches to
            # that branch as it stands. Resetting it orphaned whatever had been
            # committed on top.
            name = "at-%s" % cid
            if name not in self.branches:
                self.branches[name] = cid
            self.head = name
            return name

    def log(self, limit: int = 20, branch: Optional[str] = None) -> List[Commit]:
        with self._lock:
            name = branch or self.head
            if name not in self.branches:
                raise LatheError("unknown branch %r. Branches: %s."
                                 % (name, ", ".join(sorted(self.branches))))
            cid = self.branches[name]
            out: List[Commit] = []
            while cid is not None and len(out) < limit:
                c = self.commits[cid]
                out.append(c)
                cid = c.parent
            return out

    def geometry_changed(self, a: str, b: str) -> bool:
        return self.commit(a).geom_hash != self.commit(b).geom_hash
