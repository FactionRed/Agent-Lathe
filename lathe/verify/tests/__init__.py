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
"""
