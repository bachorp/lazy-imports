# pyright: basic

from typing import TYPE_CHECKING

from lazy_imports import LazyModule, load

if TYPE_CHECKING:
    from ..sub import three as two
    from . import reflexive
    from . import two as one
else:
    load(
        LazyModule(
            """
from . import reflexive
from ..sub import three as two
from . import two as one""",
            name=__name__,
            doc=__doc__,
        )
    )
