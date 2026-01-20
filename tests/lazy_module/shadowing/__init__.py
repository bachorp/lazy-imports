from pathlib import Path
from typing import TYPE_CHECKING

from lazy_imports import LazyModule, as_package, load

if TYPE_CHECKING:
    foo = 1
    foo = 2
else:
    load(
        LazyModule(
            *as_package(Path(__file__)),
            ("foo", 1),
            ("foo", 2),
            name=__name__,
            doc=__doc__,
        )
    )
