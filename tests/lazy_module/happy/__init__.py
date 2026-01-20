# pyright: basic

"""Hello, this is my great package."""

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from lazy_imports import LazyModule, as_package, load, module_source

__version__ = "0.1.0"


def load_lazy_module():
    load(
        LazyModule(
            ("__version__", __version__),
            ast.ImportFrom(module="argparse", names=[ast.alias(name="ArgumentParser")], level=0),
            "from sys import version_info as sysvers",
            module_source("._exports", __name__),
            *as_package(Path(__file__)),
            name=__name__,
            doc=__doc__,
        )
    )


if TYPE_CHECKING:
    from argparse import ArgumentParser
    from sys import version_info as sysvers

    from ._exports import *
else:
    load_lazy_module()
