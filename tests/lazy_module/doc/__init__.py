"""This is my doc string."""

from typing import TYPE_CHECKING

from lazy_imports import LazyModule, load

if TYPE_CHECKING:
    pass
else:
    load(
        LazyModule(
            name=__name__,
            doc=__doc__,
        )
    )
