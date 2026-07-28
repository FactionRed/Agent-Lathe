"""
Lathe -- an agent-native 3D asset authoring system.

    build     authoring kernel: primitives, booleans, selectors, assertions, bevel
    verify    rule engine and profiles
    perceive  numeric probes and structural summaries
    render    contact sheets, silhouette mattes, wireframes
    io        export
    mcp       the agent-facing tool surface

See DESIGN.md. Subpackages are imported on demand, so `import lathe` does not
pull in manifold3d, Blender or the MCP SDK.
"""

__version__ = "0.1.0"
