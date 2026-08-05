"""
Rule modules. Importing this package registers every rule in `engine.RULES`.

Grouped by what they read, not by severity, so a reader looking for "everything
that touches UVs" finds one file. Adding a rule means adding a decorated
function; adding a profile means adding a tuple of codes in profiles.py. Neither
touches engine.py.
"""

from . import budget, naming, pose, texture, topology, transform, uv  # noqa: F401

__all__ = ["budget", "naming", "pose", "texture", "topology", "transform", "uv"]
