from lazy_imports import LazyModule, load

load(
    LazyModule(
        "from sys import *",
        name=__name__,
    )
)
