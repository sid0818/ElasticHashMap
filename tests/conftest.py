"""
Shared pytest configuration.

By default, tests marked `@pytest.mark.slow` (large-scale stress tests,
e.g. the million-entry benchmark) are skipped so the everyday test suite
stays fast. Run them explicitly with:

    pytest -m slow --run-slow -v
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow / large-scale stress tests (e.g. million-entry tests)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="use --run-slow to run large-scale tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
