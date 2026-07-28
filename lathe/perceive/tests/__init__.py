"""Tests for the Lathe perception layer.

Run from the repo root:

    C:/Python312/python.exe -m pytest lathe/perceive/tests -q

The suite checks numbers against closed-form truth, not just that the code runs.
`conftest.py` puts the repo root on sys.path so `lathe` resolves as a namespace
package without an install step.
"""
