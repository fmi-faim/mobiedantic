# Copilot Instructions

## Build, test, and lint commands

- Install environments: `pixi install --all`
- Run the full test suite in the default dev target: `pixi run --environment py313 test`
- Run the CI-style coverage task: `pixi run --environment py313 cov-xml`
- Run a single test: `pixi run --environment py313 test tests/test_api.py::test_project -q`
- Run a single test file: `pixi run --environment py313 test tests/test_schemas.py -q`
- Lint: `pixi run ruff check .`
- Format: `pixi run ruff format .`
- Build distributions: `pixi run --environment build build`

Use an explicit Pixi environment for `test`/`cov-xml`; plain `pixi run test` is interactive because the task exists in multiple environments (`py310` through `py314`).

## High-level architecture

- `src/mobiedantic/generated.py` is the schema layer: generated Pydantic models for the MoBIE JSON spec. Treat it as generated code tied to the upstream schema, not the main place for handwritten logic.
- `src/mobiedantic/__init__.py` is the handwritten API layer. `Project` wraps `project.json` creation/loading/saving, and `Dataset` wraps `dataset.json` creation/loading/saving plus the higher-level helpers for images, segmentations, spots, merged grids, displays, and region tables.
- The normal write flow is: create a `Project`, call `initialize_model(...)`, create datasets with `new_dataset(...)`, initialize dataset sources/views via `Dataset` helpers, then `dataset.save()` and `project.save()`.
- Tests are split by layer:
  - `tests/test_api.py` covers the handwritten `Project`/`Dataset` API and validates written datasets on disk with `mobie.validation.validate_dataset(...)`.
  - `tests/test_schemas.py` covers the raw generated schema models, including round-tripping real upstream MoBIE dataset JSON examples.

## Key conventions

- Prefer changing `src/mobiedantic/__init__.py` for behavior changes. Only edit `generated.py` when intentionally regenerating schema models from the upstream MoBIE schema.
- Preserve schema serialization style: save JSON with `model_dump(exclude_none=True, by_alias=True)` so aliases like `is2D`, `defaultDataset`, and nested schema field names stay correct.
- Keep using dataset-root-relative paths whenever possible. `_as_source_path(...)` centralizes the `relativePath`/`absolutePath` fallback logic and should be reused instead of rebuilding path payloads manually.
- A newly initialized dataset always starts with a required `default` view configured with `uiSelectionGroup='view'` and `isExclusive=True`.
- The first dataset added to a project becomes `defaultDataset` automatically unless explicitly overridden.
- Table-backed helpers write under the dataset directory, typically `tables/<name>/default.tsv`, and schema payloads point at the table directory rather than an arbitrary external file path.
- Error text matters: tests assert specific `ValueError` messages and regex fragments, so keep existing message wording stable when refactoring.
- When changing public source/view helpers, update both the handwritten API tests and the schema/validation expectations together.
