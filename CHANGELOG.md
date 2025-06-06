# Changelog

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
