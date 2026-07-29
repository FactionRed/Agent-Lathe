"""
Lathe verification subsystem -- DESIGN.md section 7.

    Convert taste into assertions. An agent cannot have judgment, but it can
    satisfy a spec.

Usage:

    from lathe.verify import Asset, check, fixtures

    report = check(fixtures.severed_crate())
    print(report.render())        # the agent-facing text report
    report.passed                 # export gate: False when any error fired
    report.to_dict()              # the same findings as JSON

`lathe` is an implicit namespace package: this subsystem owns no code outside
`lathe/verify/`, so it adds no `lathe/__init__.py`. Run from the repository
root, or put the root on PYTHONPATH.
"""

from . import fixtures, profiles  # noqa: F401  (registers the profiles)
from .engine import (  # noqa: F401
    PROFILES,
    RULES,
    Profile,
    RuleContext,
    RuleNotApplicable,
    RuleSpec,
    check,
    get_profile,
    load_rules,
    register_profile,
    rule,
    rule_table,
    run,
)
from .findings import (  # noqa: F401
    ERROR,
    NOTE,
    WARN,
    Finding,
    Raw,
    Repair,
    Report,
    Severity,
    Skip,
)
from .mesh import (  # noqa: F401
    Asset,
    AssetMesh,
    BakeSpec,
    EmitSpec,
    TextureDecl,
    Transform,
    from_bpy_mesh,
    from_manifold,
    from_mesh,
)
from .topology import Component, Topology  # noqa: F401
from .uvmap import Island, UVLayout  # noqa: F401

__all__ = [
    "Asset",
    "AssetMesh",
    "BakeSpec",
    "Component",
    "Island",
    "EmitSpec",
    "ERROR",
    "Finding",
    "NOTE",
    "PROFILES",
    "Profile",
    "RULES",
    "Raw",
    "Repair",
    "Report",
    "RuleContext",
    "RuleNotApplicable",
    "RuleSpec",
    "Severity",
    "Skip",
    "TextureDecl",
    "Topology",
    "Transform",
    "UVLayout",
    "WARN",
    "check",
    "fixtures",
    "from_bpy_mesh",
    "from_manifold",
    "from_mesh",
    "get_profile",
    "load_rules",
    "profiles",
    "register_profile",
    "rule",
    "rule_table",
    "run",
]

__version__ = "0.1.0"
