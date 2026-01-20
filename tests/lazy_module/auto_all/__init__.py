# pyright: basic

from pathlib import Path
from typing import TYPE_CHECKING

from lazy_imports import LazyModule, as_package, load

if TYPE_CHECKING:
    foo = 42
    bar = 1
    from lazy_imports import LazyModule as _LazyModule
else:
    load(
        LazyModule(
            *as_package(Path(__file__)),
            ("foo", 42),
            ("_bar", 1),
            "from lazy_imports import LazyModule as _LazyModule",
            "from sys import version",
            name=__name__,
            doc=__doc__,
        )
    )
