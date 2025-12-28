# Copyright (c) 2025 Pascal Bachor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for working with modules."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def as_package(file: Path | str) -> Iterable[tuple[str, Any]]:
    # noqa: D205
    """Creates the attributes `__file__` and `__path__` required for a module to be a (regular) package.
    This allows to import subpackages from the appropriate locations.

    The parameter `file` should be the path to the file from which the module is loaded.
    If inside the (lazy) package's `__init__.py` file, `Path(__file__)` can be used.
    """
    path = file if isinstance(file, Path) else Path(file)
    yield ("__file__", str(path))
    yield ("__path__", (str(path.parent),))


def load(module: ModuleType) -> None:
    """Loads the module `module` by registering it in the global module store `sys.modules`."""
    sys.modules[module.__name__] = module


def module_source(name: str, package: str | None) -> str:
    """Returns the source code of the module `name` without loading the module.

    If `name` is relative, `package` must be supplied.
    """
    spec = importlib.util.find_spec(name, package)
    if spec is None:
        raise ModuleNotFoundError(
            f"could not find module {name!r}{'' if package is None else f' in package {package}'}"
        )

    return inspect.getsource(importlib.util.module_from_spec(spec))
