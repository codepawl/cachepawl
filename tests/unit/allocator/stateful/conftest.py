"""Shared hypothesis settings for the stateful allocator tests.

A per-module mypy override at ``tests.unit.allocator.stateful.*``
loosens ``disallow_any_explicit`` because Hypothesis's ``@rule`` and
``@invariant`` decorators plus its ``SearchStrategy`` generics carry
``Any`` in their public types. The rest of the strict-mode posture
(``disallow_any_generics``, ``warn_unreachable``,
``warn_unused_ignores``) stays in force for these files.

The stateful profile budgets 200 examples per test class with a 5 s
deadline so a chain of allocations + evictions on a small pool can run
without hitting Hypothesis's health checks. ``derandomize=True`` pins
the seed so a failure is reproducible from the test log alone.

The three stateful tests must fit under 60 seconds wall total per the
PR plan; today (no eviction-heavy paths shrinking poorly) the three
tests combined finish in roughly 10 seconds. If a future change pushes
them past 60 s, lower ``max_examples`` on the profile before splitting
the tests.

Hypothesis decoration choice: rule weighting is not exposed natively
in the public API, so the rule mixture is left to Hypothesis's default
exploration. Per-call probability bias is approximated by giving
``allocate`` rules narrower argument spaces (``integers(1, 8)``) than
``free`` rules (sampled from live request ids); the search reaches
both states regularly within the example budget.
"""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "stateful",
    max_examples=200,
    deadline=5000,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("stateful")
