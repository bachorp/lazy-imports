[![](https://github.com/bachorp/lazy-imports/actions/workflows/static_checks.yml/badge.svg)](https://github.com/bachorp/lazy-imports/actions/workflows/static_checks.yml)
[![](https://img.shields.io/pypi/pyversions/lazy-imports)](https://www.python.org)
[![](https://img.shields.io/pypi/v/lazy-imports.svg)](https://pypi.org/project/lazy-imports/)

This python utility package helps to create *lazy modules*.
A lazy module defers loading (some of) its attributes until these attributes are first accessed.
The module's lazy attributes in turn are attributes of other modules.
These other modules will be imported/loaded only when (and if) associated attributes are used.
A lazy import strategy can drastically reduce runtime and memory consumption.

Additionally, this package provides a utility for *optional imports* with which one can import a module globally while triggering associated import errors only at use-sites (when and if a dependency is actually required, for example in the context of a specific functionality).

[`lazy-imports`](https://pypi.org/project/lazy-imports/) is available on [the Python Package Index (PyPI)](https://pypi.org/project/lazy-imports/).

> [!IMPORTANT]
> Python's import system is highly complex and side effects are ubiquitous.
> Although employing lazy imports (in a sanely structured project) is quite safe, you should keep in mind that there are necessarily subtle differences between lazy and ordinary (eager) imports/modules.

> [!TIP]
> Using a dedicated package such as this one means that you don't have to go through all the details yourself.
> Still, we recommend to become acquainted with the basic functionality of lazy modules (such as understanding the roles of `__getattr__`, `__dir__`, and `__all__`).
> If you'd like to talk about it, feel free to open the [discussion](https://github.com/bachorp/lazy-imports/discussions) on Github.

### Table of Contents

- [`LazyModule`](#lazymodule)
  - [Example 1](#example-1)
  - [Example 2](#example-2)
- [`try_import`](#try_import)
- [`LazyImporter`](#lazyimporter)
- [History](#history)
- [Licensing](#licensing)

## [`LazyModule`](./lazy_imports/lazy_module.py)

### Example 1

Type checkers cannot reason about dynamic attributes.
Therefore, we need a separate code path (using [`typing.TYPE_CHECKING`](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)) on which regular imports are used.
By moving the imports to a separate file and using the `module_source` utility, code duplication can be avoided.

```python
# _exports.py
from absolute.package import Class, function

from .submodule import Interface as MyInterface
```

> [!IMPORTANT]
> Lazy attributes have to be proper attributes (defined in `__init__.py`) of the module they are imported from and cannot be submodules.
> In this example, if `function` is a submodule of `absolute.package`, the lazy module won't (try to) load `absolute.package.function` as a module by itself; in this case, an access of the lazy module's attribute `function` might fail.

```python
# __init__.py
"""This is my great package."""

from typing import TYPE_CHECKING
from lazy_imports import LazyModule, as_package, load, module_source

__version__ = "0.1.0"

if TYPE_CHECKING:
    from ._exports import *
else:
    # Registers the lazy module in `sys.modules`
    load(
        # The constructor takes arbitrarily many attribute declarations in various forms
        LazyModule(
            # Attributes `__file__` and `__path__` (required for importing submodules of this module)
            *as_package(__file__),
            # An ordinary (eager) attribute
            ("__version__", __version__),
            # Finds and parses the source of the submodule `_exports` (namely the file `_exports.py`)
            module_source("._exports", __name__),
            # Fully-qualified name of the module (required)
            name=__name__,
            # Docstring from the top of the module
            doc=__doc__,
        )
    )
```

### Example 2

```python
# __init__.py
import ast as _ast
from typing import TYPE_CHECKING
from lazy_imports import LazyModule as _LazyModule

# Keep outside the `not TYPE_CHECKING` branch such that your type checker does not skip this expression
_mod = LazyModule(
    *as_package(__file__),
    # A string with arbitrarily many `from <module> import <attribute>`-statements
    "from . import echo",
    # Plain type from `ast`
    ast.ImportFrom(names=[ast.alias(name="formats")], level=2),
    ast.ImportFrom(module="filters", names=[ast.alias(name="equalizer", asname="eq")], level=2),
    name=__name__,
)

if TYPE_CHECKING:
    from . import echo
    from .. import formats
    from ..filters import equalizer as eq
else:
    # NOTE: If you use this trick (instead of directly writing to `sys.modules`), you'll have to make
    #       sure that none of the attributes overlap with already defined variables (such as `_ast`).
    __getattr__, __dir__, __all__ = _mod.__getattr__, _mod.__dir__, _mod.__all__
```

## [`try_import`](./lazy_imports/try_import.py)

`try_import` is a context manager that can wrap imports of optional packages to defer exceptions.
This way you don't have to import the packages every time you call a function, but you can still import the package at the top of your module.
The context manager defers the exceptions until you actually need to use the package.
You can see an example below.

```python
from lazy_imports import try_import

with try_import() as optional_package_import:  # use `try_import` as a context manager
    import optional_package  # optional package that might not be installed

def optional_function():  # optional function that uses the optional package
    optional_package_import.check()  # check if the import was ok or raise a meaningful exception
    optional_package.some_external_function()  # use the optional package here
```

## [`LazyImporter`](./lazy_imports/v0/lazy_imports.py)

> [!TIP]
> Instead of `LazyImporter` we recommend using its successor `LazyModule`, which
>
> - allows attributes to be imported from any module (and not just submodules),
> - offers to specify imports as plain python code (which can then be sourced from a dedicated file),
> - supports `__doc__`, and
> - applies additional sanity checks (such as preventing cyclic imports).

Usage example taken from [`hpoflow/__init__.py`](https://github.com/telekom/HPOflow/blob/1b26f3b86cad607dd89a31fa9135256d956948cb/hpoflow/__init__.py):

```python
import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

from hpoflow.version import __version__


_import_structure = {
    "mlflow": [
        "normalize_mlflow_entry_name",
        "normalize_mlflow_entry_names_in_dict",
        "check_repo_is_dirty",
    ],
    "optuna": ["SignificanceRepeatedTrainingPruner"],
    "optuna_mlflow": ["OptunaMLflow"],
    "optuna_transformers": ["OptunaMLflowCallback"],
    "utils": ["func_no_exception_caller"],
}

# Direct imports for type-checking
if TYPE_CHECKING:
    from hpoflow.mlflow import (  # noqa: F401
        check_repo_is_dirty,
        normalize_mlflow_entry_name,
        normalize_mlflow_entry_names_in_dict,
    )
    from hpoflow.optuna import SignificanceRepeatedTrainingPruner  # noqa: F401
    from hpoflow.optuna_mlflow import OptunaMLflow  # noqa: F401
    from hpoflow.optuna_transformers import OptunaMLflowCallback  # noqa: F401
    from hpoflow.utils import func_no_exception_caller  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"__version__": __version__},
    )
```

## History

This project has previously been maintained by the [One Conversation](https://welove.ai/) team of [Deutsche Telekom AG](https://www.telekom.com/).
It is based on [`_LazyModule`](https://github.com/huggingface/transformers/blob/e218249b02465ec8b6029f201f2503b9e3b61feb/src/transformers/file_utils.py#L1945) from [HuggingFace](https://huggingface.co/) and [`try_import()`](https://github.com/optuna/optuna/blob/1f92d496b0c4656645384e31539e4ee74992ff55/optuna/_imports.py#L89) from the [Optuna framework](https://optuna.readthedocs.io/).
Many thanks to HuggingFace for [your consent](https://github.com/huggingface/transformers/issues/12861#issuecomment-886712209)
and to Optuna for [your consent](https://github.com/optuna/optuna/issues/2776#issuecomment-874614137) to publish it as a standalone package 🤗 ♥.

In December 2024 responsibility was transferred to [Pascal Bachor](https://github.com/bachorp).

## Licensing

Copyright (c) 2024-2025 Pascal Bachor<br/>
Copyright (c) 2021 [Philip May](https://philipmay.org/), [Deutsche Telekom AG](https://www.telekom.com/)<br/>
Copyright (c) 2020, 2021 [The HuggingFace Team](https://huggingface.co/)<br/>
Copyright (c) 2018 Preferred Networks, Inc.

Licensed under the [Apache License, Version 2.0](https://github.com/bachorp/lazy-imports/blob/HEAD/LICENSE) (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
