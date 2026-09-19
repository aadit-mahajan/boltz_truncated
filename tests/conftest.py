"""Shared fixtures for the optimization tests."""

import os

import pytest

from boltz import opt


@pytest.fixture(autouse=True)
def restore_profile():
    """Put the process-global profile back, exactly, after every test.

    Both halves have to be restored: `configure` falls back to the environment
    for whichever argument it is not given, so replaying only the profile would
    leave a previous test's `--disable_opt` in force.
    """
    profile = opt.active_profile()
    disabled = tuple(
        name for name in os.environ.get("BOLTZ_OPT_DISABLE", "").split(",") if name
    )
    yield
    opt.configure(profile, disabled)
