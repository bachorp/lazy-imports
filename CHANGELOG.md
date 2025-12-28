# Changelog

## 1.2.0

Bump python to 3.10 and modernize.

## 1.1.0

Adds support for Python 3.14.

## 1.0.1

### Bug Fixes

- Make sure not to include additional packages in dist (https://github.com/bachorp/lazy-imports/issues/4)

### Documentation

- fix link to Philip May (https://github.com/bachorp/lazy-imports/pull/1) by @PhilipMay
- utilize dedicated license parameter (https://github.com/bachorp/lazy-imports/pull/5/commits/66200c1f82a56f6aed98f2456ebe59c9a0406856)

## 1.0.0

`v1` introduces a new class `LazyModule`, an improved version of `LazyImporter`, which

- allows attributes to be imported from any module (and not just submodules),
- offers to specify imports as plain python code (which can then be sourced from a dedicated file),
- supports `__doc__`, and
- applies additional sanity checks (such as preventing cyclic imports).

## 0.4.0

- Bump minimum version of Python to `3.9`
- Raise `ValueError` upon instantiating `LazyImporter` with duplicate attributes
- Include `extra_objects` in `__all__` and `__reduce__`
- Add type annotations to `LazyImporter(...)`
