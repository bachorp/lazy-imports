from pathlib import Path
from typing import TYPE_CHECKING

from lazy_imports import LazyModule, as_package, load

if TYPE_CHECKING:
    pass
else:
    load(
        LazyModule(
            *as_package(Path(__file__)),
            ("foo", 1),
            ("from sys import version as bar"),
            name=__name__,
            doc=__doc__,
        )
    )
