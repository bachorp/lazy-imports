# pyright: basic

from pathlib import Path
from typing import TYPE_CHECKING

from lazy_imports import LazyModule, as_package, load


if TYPE_CHECKING:
    from .. import symmetric
    from ..sub2 import one as three
else:
    load(
        LazyModule(
            *as_package(Path(__file__)),
            "from ..sub2 import one as three",
            "from .. import symmetric",
            name=__name__,
            doc=__doc__,
        )
    )
