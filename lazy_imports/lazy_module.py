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

"""LazyModule and associated types."""

import ast
import importlib
import itertools
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterable, Union


# Can be improved with later versions of Python:
# 3.10+: Union -> |, if/elif -> match, TypeAlias
# 3.12+: type statement

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing import NoReturn

    def assert_never(arg: NoReturn) -> NoReturn:  # noqa: D103; pylint: disable=missing-function-docstring
        value = repr(arg)
        if len(value) > 100:
            value = value[:100] + "..."
        raise AssertionError(f"Expected code to be unreachable, but got: {value}")


Statement = Union[tuple[str, Any], ast.ImportFrom]


def _parse(statement_or_code: Union[str, "Statement"]) -> Iterable["Statement"]:
    if isinstance(statement_or_code, str):
        for stmt in ast.parse(statement_or_code).body:
            if isinstance(stmt, ast.ImportFrom):
                yield stmt
            elif isinstance(stmt, ast.Import):
                raise ValueError(f"statements of type {ast.Import.__qualname__} are not supported")
            else:
                raise ValueError(
                    f"expected parsed statement to be of type {ast.ImportFrom.__qualname__} but got {type(stmt).__qualname__}"  # noqa: E501
                )
    else:
        yield statement_or_code


@dataclass
class _Immediate:
    value: Any

    def __str__(self) -> str:
        return f"object of type {type(self.value).__qualname__}"


@dataclass
class _AttributeImport:
    level: int
    module: Union[str, None]
    name: str

    def module_relatively(self) -> str:  # noqa: D103; pylint: disable=missing-function-docstring
        return f"{'.' * (self.level)}{self.module or ''}"

    def __str__(self) -> str:
        return f"attribute {self.name!r} imported from module {self.module_relatively()}"


_Deferred = _AttributeImport
_AttributeValue = Union[_Immediate, _Deferred]


@dataclass
class _Attribute:
    name: str
    value: _AttributeValue


def _to_attributes(statement: "Statement") -> Iterable[_Attribute]:
    if isinstance(statement, tuple):
        yield _Attribute(name=statement[0], value=_Immediate(value=statement[1]))
    elif isinstance(statement, ast.ImportFrom):  # pyright: ignore[reportUnnecessaryIsInstance]
        for name in statement.names:
            if name.name == "*":
                raise ValueError(f"cannot lazily perform a wildcard import (from module {statement.module})")

            yield _Attribute(
                name=name.asname or name.name,
                value=_AttributeImport(module=statement.module, name=name.name, level=statement.level),
            )
    else:
        assert_never(statement)


class ShadowingWarning(UserWarning):
    """This warning signals that an attribute is shadowing some other attribute of a lazy module."""


class LazyModule(ModuleType):
    """A module whose attributes, if they are defined to be attributes of other modules, are resolved lazily in the sense that loading the corresponding module is deferred until the attribute is first accessed.

    Constructor arguments:
        - `statement_or_code` *(repeated positional)* - Definition of the module's main attributes. Each element is either
            - a `str` to be parsed as python code consisting of `from <module> import <attribute>` statements,
            - an instance of `ast.ImportFrom`, or
            - a tuple `(<name>, <value>)` constituting a plain (non-lazy) attribute.
        - `name` *(required)* - The module's name (attribute `__name__`).
        - `doc` - The module's docstring (attribute `__doc__`).
        - `auto_all` *(default: `True`)* - Whether to automatically generate and include the attribute `__all__` if not given.
        (This is required to support wildcard imports of deferred attributes. Note that a wildcard import causes immediate resolution of the imported attributes.)

    For examples and additional information please visit the project's homepage at https://github.com/bachorp/lazy-imports/.
    """  # noqa: E501

    def __init__(
        self,
        *statement_or_code: Union[str, Union[ast.ImportFrom, tuple[str, Any]]],
        name: str,
        doc: Union[str, None] = None,
        auto_all: bool = True,
    ) -> None:
        super().__init__(name, doc)
        self.__deferred_attrs: dict[str, _Deferred] = {}
        self.__resolving: dict[str, object] = {}

        attrs: dict[str, _AttributeValue] = {}
        for attr in itertools.chain(*map(_to_attributes, itertools.chain(*map(_parse, statement_or_code)))):
            if (existing := attrs.get(attr.name)) is not None:
                warnings.warn(
                    ShadowingWarning(f"{attr.name} ({attr.value}) shadows {existing} in lazy module {self.__name__}")
                )

            attrs[attr.name] = attr.value

        for name, value in attrs.items():  # pylint: disable=redefined-argument-from-local
            if hasattr(self, name):
                raise ValueError(f"not allowed to override reserved attribute {name!r} (with {value})")

            if isinstance(value, _Immediate):
                setattr(self, name, value.value)
            elif isinstance(value, _AttributeImport):  # pyright: ignore[reportUnnecessaryIsInstance]
                self.__deferred_attrs[name] = value
            else:
                assert_never(value)

        # NOTE: Explicit __all__ is required because otherwise potential wildcard imports will use the actual
        #       attributes, ignoring __dir__.
        if auto_all and "__all__" not in dir(self):
            setattr(self, "__all__", (*filter(lambda name: not name.startswith("_"), dir(self)),))

    def __dir__(self) -> Iterable[str]:
        # NOTE: If `sub` is a deferred attribute import when the submodule `.sub` is loaded, the import system will
        #       register `sub` as an attribute (of its parent module) such that `sub` appears in both `super.__dir__()`
        #       and `self.__deferred_attrs.keys()`. It will then remain deferred indefinitely.
        #       That's why we have to purge duplicates here.
        return set(itertools.chain(super().__dir__(), self.__deferred_attrs.keys()))

    def __getattr__(self, name: str) -> Any:
        if self.__resolving.setdefault(name, (o := object())) is not o:  # setdefault is atomic
            raise ImportError(
                f"cannot resolve attribute {name!r} of lazy module {self.__name__} whose resolution is already pending (most likely due to a circular import)"  # noqa: E501
            )

        try:
            target = self.__deferred_attrs.get(name)
            if target is None:
                raise AttributeError(f"lazy module {self.__name__} has no attribute {name!r}")

            try:
                value = getattr(importlib.import_module(target.module_relatively(), self.__name__), target.name)
            except Exception as e:
                if sys.version_info >= (3, 11):
                    e.add_note(  # pylint: disable=no-member
                        f"resolving attribute {name!r} ({target}) of lazy module {self.__name__}"
                    )

                raise

            setattr(self, name, value)
            self.__deferred_attrs.pop(name)
            return value
        finally:
            self.__resolving.pop(name)
