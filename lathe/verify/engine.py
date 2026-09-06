"""
Lathe DESIGN.md section 7 -- the rule engine.

Requirement: rules are data, not control flow. Section 7 says "rules are grouped
into profiles selected by `emit`", and lists four additional profiles
(`character_skinned`, `hard_surface_subdiv`, `film_usd`, `mobile_vr`). If adding
one of those meant editing the engine, the engine would be an if-chain wearing a
hat. All four exist now, and this module never learned any of their names.

The shape:

    @rule("E107", ERROR, "Triangle count over declared budget",
          needs=("budget",), defaults={"tolerance": 0})
    def e107(ctx): yield ctx.finding(...)

    RULES["E107"]  ->  RuleSpec

A Profile is a tuple of codes plus per-code parameter and severity overrides.
`run()` resolves codes against the registry, builds one `RuleContext` per asset
(so derived topology is computed once, not once per rule), and collects
findings. Nothing in this module knows what E107 means.

Three engine-level behaviours the rules rely on:

  * `ctx.not_applicable(reason)` -- a rule with no declaration to compare
    against reports itself as *not checked*. It never passes silently. A false
    pass and a false failure are both poison in an agent feedback loop; the
    difference is that a false pass is invisible.
  * caching -- `ctx.topology` and `ctx.uv()` are memoised per asset.
  * determinism -- findings sort by (severity desc, code, entity), so two runs
    of the same asset produce identical text, which is what makes reports
    diffable across a history DAG (section 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .findings import Finding, Repair, Report, Severity, Skip
from .mesh import Asset, AssetMesh, EmitSpec
from .topology import Topology
from .uvmap import UVLayout

# ------------------------------------------------------------------ registry


class RuleNotApplicable(Exception):
    """Raised by a rule that has nothing to check. Becomes a Skip, not a pass."""


@dataclass(frozen=True)
class RuleSpec:
    code: str
    severity: Severity
    title: str
    check: Callable[["RuleContext"], Iterable[Finding]]
    needs: Tuple[str, ...] = ()  # documentation of required declarations
    defaults: Mapping[str, Any] = field(default_factory=dict)
    approximation: Optional[str] = None  # honest note, surfaced in the README
    doc: str = ""

    @property
    def is_approximate(self) -> bool:
        return self.approximation is not None


RULES: Dict[str, RuleSpec] = {}


def rule(
    code: str,
    severity: Severity,
    title: str,
    needs: Sequence[str] = (),
    defaults: Optional[Mapping[str, Any]] = None,
    approximation: Optional[str] = None,
):
    """Register a rule. Import of the module is the registration."""

    def wrap(fn: Callable[["RuleContext"], Iterable[Finding]]) -> Callable:
        if code in RULES:
            raise ValueError("duplicate rule code %s" % code)
        RULES[code] = RuleSpec(
            code=code,
            severity=severity,
            title=title,
            check=fn,
            needs=tuple(needs),
            defaults=dict(defaults or {}),
            approximation=approximation,
            doc=(fn.__doc__ or "").strip(),
        )
        return fn

    return wrap


def load_rules() -> Dict[str, RuleSpec]:
    """Import every rule module once, so the registry is populated."""
    from . import rules as _rules  # noqa: F401  (side effect: registration)

    return RULES


# ------------------------------------------------------------------- profile


@dataclass(frozen=True)
class Profile:
    """A named rule set. Pure data -- adding one touches no engine code."""

    name: str
    rules: Tuple[str, ...]
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    severity: Mapping[str, Severity] = field(default_factory=dict)
    description: str = ""

    def with_overrides(self, **kw) -> "Profile":
        return replace(self, **kw)


PROFILES: Dict[str, Profile] = {}


def register_profile(profile: Profile) -> Profile:
    PROFILES[profile.name] = profile
    return profile


def get_profile(name: Union[str, Profile]) -> Profile:
    if isinstance(name, Profile):
        return name
    load_profiles()
    if name not in PROFILES:
        raise KeyError(
            "unknown profile %r; registered: %s" % (name, ", ".join(sorted(PROFILES)))
        )
    return PROFILES[name]


def load_profiles() -> Dict[str, Profile]:
    from . import profiles as _profiles  # noqa: F401  (side effect: registration)

    return PROFILES


# ------------------------------------------------------------------- context


class RuleContext:
    """Everything a rule may read, and nothing it may mutate."""

    def __init__(self, asset: Asset, spec: RuleSpec, params: Mapping[str, Any],
                 severity: Severity, cache: Optional[Dict[str, Any]] = None) -> None:
        self.asset = asset
        self.spec = spec
        self.params = dict(params)
        self.severity = severity
        self._cache = cache if cache is not None else {}

    # -- shortcuts ---------------------------------------------------------

    @property
    def mesh(self) -> AssetMesh:
        return self.asset.mesh

    @property
    def emit(self) -> EmitSpec:
        return self.asset.spec

    @property
    def name(self) -> str:
        return self.asset.mesh.name

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    # -- cached derivations ------------------------------------------------

    @property
    def topology(self) -> Topology:
        if "topology" not in self._cache:
            self._cache["topology"] = Topology(self.mesh, weld=self.emit.weld_tolerance)
        return self._cache["topology"]

    def uv(self, set_name: Optional[str] = None) -> UVLayout:
        """The declared bake uv set by default. Raises RuleNotApplicable when
        the asset has no such set -- absence of UVs is not a UV violation."""
        wanted = set_name or (self.emit.bake.uv_set if self.emit.bake else None)
        if wanted is None:
            wanted = next(iter(sorted(self.mesh.uv_sets)), None)
        if wanted is None or wanted not in self.mesh.uv_sets:
            self.not_applicable(
                "asset declares no uv set %s"
                % ("%r" % wanted if wanted else "at all")
            )
        key = "uv:%s" % wanted
        if key not in self._cache:
            self._cache[key] = UVLayout(self.mesh, wanted)
        return self._cache[key]

    # -- outputs -----------------------------------------------------------

    def not_applicable(self, reason: str) -> None:
        raise RuleNotApplicable(reason)

    def finding(
        self,
        entity: str,
        evidence: str,
        repair: Repair,
        summary: Optional[str] = None,
        where: Sequence[Tuple[str, Union[str, Sequence[str]]]] = (),
        data: Optional[Mapping[str, Any]] = None,
    ) -> Finding:
        """Build a Finding with the code, severity and title of this rule."""
        return Finding(
            code=self.spec.code,
            severity=self.severity,
            summary=summary or self.spec.title,
            entity=entity,
            evidence=evidence,
            repair=repair,
            where=tuple(where),
            data=dict(data or {}),
            rule=self.spec.title,
        )


# ----------------------------------------------------------------- execution


def run(
    asset: Asset,
    profile: Union[str, Profile, None] = None,
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Report:
    """Evaluate one profile against one asset. The only entry point."""
    load_rules()
    prof = get_profile(profile if profile is not None else asset.spec.profile)
    cache: Dict[str, Any] = {}
    findings: list = []
    skipped: list = []

    for code in prof.rules:
        if code not in RULES:
            raise KeyError(
                "profile %r selects unregistered rule %r; registered: %s"
                % (prof.name, code, ", ".join(sorted(RULES)))
            )
        spec = RULES[code]
        params = dict(spec.defaults)
        params.update(prof.params.get(code, {}))
        params.update((overrides or {}).get(code, {}))
        sev = prof.severity.get(code, spec.severity)
        ctx = RuleContext(asset, spec, params, sev, cache)
        try:
            produced = list(spec.check(ctx) or ())
        except RuleNotApplicable as exc:
            skipped.append(Skip(code=code, reason=str(exc)))
            continue
        findings.extend(produced)

    findings.sort(key=lambda f: (-int(f.severity), f.code, f.entity, f.summary))
    return Report(
        asset=asset.mesh.name,
        profile=prof.name,
        findings=tuple(findings),
        skipped=tuple(skipped),
    )


def check(asset: Asset, profile: Union[str, Profile, None] = None, **kw) -> Report:
    """Section 9's `check(profile)`."""
    return run(asset, profile, **kw)


def rule_table(profile: Union[str, Profile, None] = None) -> Iterator[RuleSpec]:
    """Registered rules, in code order. Used by the README generator and by
    tests that assert the profile and the documentation agree."""
    load_rules()
    if profile is None:
        codes = sorted(RULES)
    else:
        codes = list(get_profile(profile).rules)
    for code in codes:
        yield RULES[code]
