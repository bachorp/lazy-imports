# pyright: basic

import pytest

from lazy_imports import ShadowingWarning


def test_happy():
    """Tests general functionality of LazyModule."""
    import happy

    assert happy.__doc__ == "Hello, this is my great package."
    assert happy.__version__ == "0.1.0"
    assert happy.f() == 6
    assert happy.g == happy.h
    x, y = happy.ArgumentParser, happy.sysvers
    from argparse import ArgumentParser
    from sys import version_info

    assert (x, y) == (ArgumentParser, version_info)


def test_doc():
    """Tests the functionality of doc/__doc__."""
    import doc

    assert doc.__doc__ == "This is my doc string."


def test_auto_all_():
    """Tests the functionality of auto____all__."""
    import auto_all

    assert set(getattr(auto_all, "__all__")) == set(["foo", "version"])


def test_shadowing():
    """Tests whether attribute shadowing works, including issuing of ShadowingWarning."""
    with pytest.warns(ShadowingWarning, match="foo"):
        import shadowing


def test___dir__():
    """Tests whether dir() includes both deferred and proper attributes of a lazy module."""
    import dir_

    assert set(filter(lambda x: not x.startswith("__") and not x.startswith("_LazyModule__"), dir(dir_))) == set(
        ("foo", "bar")
    )


def test_cycle():
    """Tests whether ImportError is raised in case of a circular import/attribute access through a lazy module."""
    import cycle

    with pytest.raises(
        ImportError,
        match="circular",
    ):
        cycle.reflexive
    with pytest.raises(
        ImportError,
        match="circular",
    ):
        cycle.symmetric
    with pytest.raises(
        ImportError,
        match="circular",
    ):
        from cycle import sub

        sub.three


def test_submodule():
    """Tests whether submodules of a lazy module are correctly registered with their parent on import."""
    import submodule as x
    import submodule.sub as y

    assert getattr(x, "sub") == y


def test_fail():
    """Checks behavior on invalid arguments passed to LazyModule."""
    with pytest.raises(ValueError, match="ImportFrom"):
        import fail
    with pytest.raises(ValueError, match="ImportFrom"):
        from fail import sub
