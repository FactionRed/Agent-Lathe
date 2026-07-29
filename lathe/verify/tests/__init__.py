"""
Tests for the verifier.

    python -m lathe.verify.tests            # run everything
    python lathe/verify/tests/run.py -v     # same, verbose

Three files, three claims:

  test_rules.py   every rule fires on a broken fixture with correct numeric
                  evidence, and no rule fires on a clean one
  test_report.py  the report keeps the shape section 7 specifies, in text and
                  in JSON, for every finding the suite can produce
  test_engine.py  rules are data: profiles select, parametrise and re-grade
                  them, and a new profile needs no engine change

The registry is loaded here, at import, and that is not decoration.

`RULES` is populated lazily -- `check`, `run` and `rule_table` each call
`load_rules()` themselves, so the public surface is fine, but a bare
`import lathe.verify` leaves the dict empty. The suite used to depend on
`TestRegistry.test_rules_are_registered_by_import` calling `load_rules()` and
mutating the shared dict in place, which meant every other test needed that one
to have run first.

It worked under `unittest`, by luck: unittest sorts class names, and `TestRegistry`
sorts sixth while four classes that call `run()` sort ahead of it and populate the
registry as a side effect. `pytest` preserves *definition* order, where
`TestRegistry` is first -- so it asserted against an empty registry and 241 tests
failed in cascade. The suite passed one way and collapsed the other, having never
been run the second way.
"""

from ..engine import load_rules

load_rules()
