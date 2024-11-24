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
import os
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Collection, Dict, Iterable, List, Tuple, Union  # TODO: Get rid of Union in 3.10+


@dataclass
class _Immediate:
    value: Any

    def __str__(self) -> str:
        return f"object of type '{type(self.value).__name__}'"


@dataclass
class _Deferred:
    pass


@dataclass
class _ModuleImport(_Deferred):
    name: str

    def __str__(self) -> str:
        return f"import of module '{self.name}'"


@dataclass
class _AttributeImport(_Deferred):
    module: Union[str, None]
    name: str
    level: int

    def expand(self) -> str:
        return "." * (self.level) + (self.module or "")

    def __str__(self) -> str:
        return f"attribute '{self.name}' imported from module '{self.expand()}'"


_Attr = Union[_Immediate, _ModuleImport, _AttributeImport]


@dataclass
class _Subattribute:
    name: str
    value: _Attr

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


class _Submodule(List[_Subattribute]):
    def __str__(self) -> str:
        return f"submodule {{{', '.join(map(lambda s: f'{s.name}: {s.value}', self))}}}"


def _parse(
    s: Union[str, Tuple[str, Any], ast.Import, ast.ImportFrom]
) -> Iterable[Union[Tuple[str, Any], ast.Import, ast.ImportFrom]]:
    if isinstance(s, str):
        for stmt in ast.parse(s).body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                yield stmt
            else:
                raise ValueError(
                    f"expected parsed statement to be of type {ast.Import.__name__} or {ast.ImportFrom.__name__} but got {type(stmt).__name__}"  # noqa: E501
                )
    else:
        yield s


def _to_attributes(stmt: Union[Tuple[str, Any], ast.Import, ast.ImportFrom]) -> Iterable[Tuple[str, _Attr]]:
    if isinstance(stmt, ast.Import):
        for name in stmt.names:
            yield (name.asname or name.name, _ModuleImport(name=name.name))
    elif isinstance(stmt, ast.ImportFrom):
        for name in stmt.names:
            if name.name == "*":
                raise ValueError(f"cannot lazily perform a wildcard import (from '{stmt.module}')")

            yield (name.asname or name.name, _AttributeImport(module=stmt.module, name=name.name, level=stmt.level))
    elif isinstance(stmt, tuple):  # type: ignore [reportUnnecessaryIsInstance]
        yield (stmt[0], _Immediate(value=stmt[1]))
    else:
        assert False


Statement = Union[Tuple[str, Any], ast.Import, ast.ImportFrom]


class ShadowingWarning(UserWarning):
    """TODO."""


class LazyModule(ModuleType):
    """TODO."""

    def __init__(
        self,
        *statements: Union[str, Statement],
        name: str,
        doc: Union[str, None] = None,
        unsafe_overrides: Collection[str] = frozenset(),
    ) -> None:
        super().__init__(name, doc)
        self.__deferred_attrs: Dict[str, Union[_ModuleImport, _AttributeImport]] = {}
        self.__resolving: Dict[str, object] = {}

        merged_attributes: Dict[str, Union[_Attr, _Submodule]] = {}
        for name, value in itertools.chain(*map(_to_attributes, itertools.chain(*map(_parse, statements)))):
            name, sub_name = name.split(".", maxsplit=1) if "." in name else (name, None)
            existing = merged_attributes.get(name)

            def shadow() -> None:
                warnings.warn(
                    ShadowingWarning(f"'{name}' ({value}) shadows {existing} in lazy module {self.__name__}")
                )
                merged_attributes.pop(name)

            if sub_name is None:
                if existing is not None:
                    shadow()

                merged_attributes[name] = value
            else:
                if type(existing) not in (_Submodule, type(None)):  # TODO: Replace with NoneType in 3.10+
                    shadow()

                merged_attributes.setdefault(
                    name, _Submodule()
                ).append(  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue, union-attr]
                    _Subattribute(name=sub_name, value=value)
                )

        for name, value in merged_attributes.items():  # type: ignore[assignment]
            if hasattr(self, name) and name not in unsafe_overrides:
                raise ValueError(f"not allowed to override reserved attribute '{name}' (with {value})")
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
                f"cannot resolve attribute '{name}' of lazy module '{self.__name__}' whose resolution is already pending (most likely due to a circular import)"  # noqa: E501
            )

        try:
            target = self.__deferred_attrs.get(name)
            if target is None:
                raise AttributeError(f"lazy module '{self.__name__}' has no attribute '{name}'")

            try:
                if isinstance(target, _ModuleImport):
                    value = importlib.import_module(target.name)
                elif isinstance(target, _AttributeImport):  # type: ignore [reportUnnecessaryIsInstance]
                    value = getattr(importlib.import_module(target.expand(), self.__name__), target.name)
                else:
                    assert False

            except Exception as e:
                if sys.version_info >= (3, 11):
                    e.add_note(f"resolving attribute '{name}' ({target}) of lazy module '{self.__name__}'")
                raise

            setattr(self, name, value)
            self.__deferred_attrs.pop(name)
            return value

        finally:
            self.__resolving.pop(name)


def as_package(file: str) -> Iterable[Statement]:
    """TODO."""
    yield ("__file__", file)
    yield ("__path__", (os.path.dirname(file),))


def register(module: LazyModule) -> None:
    """TODO."""
    sys.modules[module.__name__] = module
