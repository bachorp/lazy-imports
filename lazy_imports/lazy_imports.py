# Copyright (c) 2024 Pascal Bachor
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

"""TODO."""

# pylint: disable=missing-function-docstring,redefined-argument-from-local,cell-var-from-loop

import ast
import importlib
import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Collection, Dict, Iterable, List, Tuple, Union  # TODO: Get rid of Union in 3.10+


@dataclass
class _Immediate:
    value: Any

    def __str__(self) -> str:
        return f"object of type {type(self.value).__name__}"


@dataclass
class _ModuleImport:
    name: str

    def __str__(self) -> str:
        return f"import of module {self.name}"


@dataclass
class _AttributeImport:
    level: int
    module: Union[str, None]
    name: str

    def module_relatively(self) -> str:
        return "." * (self.level) + (self.module or "")

    def __str__(self) -> str:
        return f"attribute {self.name} imported from module {self.module_relatively()}"


# TODO: Declare as TypeAlias in 3.10+ and use type statement in 3.12+
_Deferred = Union[_ModuleImport, _AttributeImport]
_AttributeValue = Union[_Immediate, _Deferred]


@dataclass
class _Attribute:
    name: str
    value: _AttributeValue

    def to_statement(self) -> "Statement":
        if isinstance(self.value, _Immediate):
            return (self.name, self.value.value)

        if isinstance(self.value, _ModuleImport):
            return ast.Import([ast.alias(name=self.value.name, asname=self.name)])

        if isinstance(self.value, _AttributeImport):  # type: ignore [reportUnnecessaryIsInstance]
            return ast.ImportFrom(
                module=self.value.module,
                names=[ast.alias(name=self.value.name, asname=self.name)],
                level=self.value.level,
            )

        assert False


class _Submodule(List[_Attribute]):
    def __str__(self) -> str:
        return f"submodule {{{', '.join(map(lambda s: f'{s.name}: {s.value}', self))}}}"


def _parse(statement_or_code: Union[str, "Statement"]) -> Iterable["Statement"]:
    if isinstance(statement_or_code, str):
        for stmt in ast.parse(statement_or_code).body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                yield stmt
            else:
                raise ValueError(
                    f"expected parsed statement to be of type {ast.Import.__name__} or {ast.ImportFrom.__name__} but got {type(stmt).__name__}"  # noqa: E501
                )
    else:
        yield statement_or_code


def _to_attributes(statement: "Statement") -> Iterable[_Attribute]:
    if isinstance(statement, tuple):
        yield _Attribute(name=statement[0], value=_Immediate(value=statement[1]))
    elif isinstance(statement, ast.Import):
        for name in statement.names:
            yield _Attribute(name=name.asname or name.name, value=_ModuleImport(name=name.name))
    elif isinstance(stmt, ast.ImportFrom):  # type: ignore [reportUnnecessaryIsInstance]
        for name in statement.names:
            if name.name == "*":
                raise ValueError(f"cannot lazily perform a wildcard import (from module {statement.module})")

            yield _Attribute(
                name=name.asname or name.name,
                value=_AttributeImport(module=statement.module, name=name.name, level=statement.level),
            )
    else:
        assert False


# TODO: Declare as TypeAlias in 3.10+ and use type statement in 3.12+
Statement = Union[Tuple[str, Any], ast.Import, ast.ImportFrom]


class ShadowingWarning(UserWarning):
    """This warning signals that an attribute is shadowing some other attribute of a lazy module."""


class LazyModule(ModuleType):
    """A module whose attributes, which can be modules or attributes of other modules, are resolved lazily in the sense that loading the corresponding module is deferred until an attribute is first accessed.

    Arguments:
        - `*statements`: Definition of the module's main attributes. Each element is either
            - a `str` to be parsed as a sequence of import statements,
            - an instance of `ast.Import`,
            - an instance of `ast.ImportFrom`, or
            - a tuple `(name, value)` constituting a plain (non-lazy) attribute.
        - `name` (required): The module's name (attribute `__name__`).
        - `doc`: The module's docstring (attribute `__doc__`).
        - `unsafe_overrides`: Existing attributes (e.g. `__dir__`) that are allowed to be overridden.
    """

    def __init__(
        self,
        *statements_or_code: Union[
            str, Union[ast.Import, ast.ImportFrom, Tuple[str, Any]]  # spell out Statement for documentation
        ],
        name: str,
        doc: Union[str, None] = None,
        unsafe_overrides: Collection[str] = frozenset(),
    ) -> None:
        super().__init__(name, doc)
        self.__deferred_attrs: Dict[str, Union[_ModuleImport, _AttributeImport]] = {}
        self.__resolving: Dict[str, object] = {}

        merged_attributes: Dict[str, Union[_AttributeValue, _Submodule]] = {}
        for attr in itertools.chain(*map(_to_attributes, itertools.chain(*map(_parse, statements_or_code)))):
            name, sub_name = name.split(".", maxsplit=1) if "." in attr.name else (attr.name, None)
            existing = merged_attributes.get(name)

            def shadow() -> None:
                warnings.warn(ShadowingWarning(f"{name} ({value}) shadows {existing} in lazy module {self.__name__}"))
                merged_attributes.pop(name)

            if sub_name is None:
                if existing is not None:
                    shadow()

                merged_attributes[name] = attr.value
            else:
                if type(existing) not in (_Submodule, type(None)):  # TODO: Replace with NoneType in 3.10+
                    shadow()

                merged_attributes.setdefault(
                    name, _Submodule()
                ).append(  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue, union-attr]
                    _Attribute(name=sub_name, value=attr.value)
                )

        for name, value in merged_attributes.items():  # type: ignore[assignment]
            if hasattr(self, name) and name not in unsafe_overrides:
                raise ValueError(f"not allowed to override reserved attribute {name} (with {value})")
            if isinstance(value, _Immediate):
                setattr(self, name, value.value)
            elif isinstance(value, _Deferred):
                self.__deferred_attrs[name] = value
            elif isinstance(value, _Submodule):  # type: ignore [reportUnnecessaryIsInstance]
                setattr(
                    self, name, LazyModule(*map(lambda sub: sub.to_statement(), value), name=f"{self.__name__}.{name}")
                )
            else:
                assert False

    def __dir__(self) -> Iterable[str]:
        return itertools.chain(super().__dir__(), self.__deferred_attrs.keys())

    def __getattr__(self, name: str) -> Any:
        if self.__resolving.setdefault(name, (o := object())) is not o:  # setdefault is atomic
            raise ImportError(
                f"cannot resolve attribute {name} of lazy module {self.__name__} whose resolution is already pending (most likely due to a circular import)"  # noqa: E501
            )

        try:
            target = self.__deferred_attrs.get(name)
            if target is None:
                raise AttributeError(f"lazy module {self.__name__} has no attribute {name}")

            try:
                if isinstance(target, _ModuleImport):
                    value = importlib.import_module(target.name)
                elif isinstance(target, _AttributeImport):  # type: ignore [reportUnnecessaryIsInstance]
                    value = getattr(importlib.import_module(target.module_relatively(), self.__name__), target.name)
                else:
                    assert False
            except Exception as e:
                if sys.version_info >= (3, 11):  # TODO: Remove in 3.11+
                    e.add_note(  # pylint: disable=no-member
                        f"resolving attribute {name} ({target}) of lazy module {self.__name__}"
                    )
                raise

            setattr(self, name, value)
            self.__deferred_attrs.pop(name)
            return value
        finally:
            self.__resolving.pop(name)


def as_package(file: Path) -> Iterable[Statement]:
    """Creates the attributes `__file__` and `__path__` required for a module being a proper package.
    This allows to import subpackges from the appropriate locations.

    The parameter `file` should be the path to the file from which the module is loaded.
    """
    yield ("__file__", str(file))
    yield ("__path__", (str(file.parent.name),))


def load(module: ModuleType) -> None:
    """Loads the module `module` by registering it in the global module store `sys.modules`."""
    sys.modules[module.__name__] = module
