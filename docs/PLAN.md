# mobiedantic - data models for MoBIE projects and datasets

A lean utility package for creating and manipulating MoBIE project.json and dataset.json files,
backed by pydantic models for the project and dataset specs,
offering API to manage (image, segmentation, ...) sources and views in MoBIE datasets.


## Background

When writing MoBIE project JSON from code, we can use `mobie-utils-python` to manage
our data structure and validate the output. However, `mobie-utils-python` comes with
a huge dependency footprint. Some of its dependencies are exclusively distributed via
conda-forge, but not available from PyPI.

This project aims to have minimal dependencies and to be easily installable on any platform.
We currently deploy to PyPI only, but aim to publish on conda-forge as well.

### Reference projects

* https://github.com/mobie/mobie-utils-python
* https://github.com/mobie/mobie-viewer-fiji


## Environment management - pixi

We use `pixi` tasks to run tests, build and lint code.

Run tests:
```
pixi run test
```

In a specific environment:
```
pixi run -e py311 test
```

With additional `pytest` options:
```
pixi run test -vv -s -o log_cli=1 --basetemp=./tmp
```

Build package:
```
pixi run build
```


### Onboarding

```
git clone <repo> && cd mobiedantic
pixi install --all                 # installs all environments
pixi run test
```

No `pip`, no `venv`, no `poetry`. Contributors install pixi once (`curl -fsSL https://pixi.sh/install.sh | bash`) and that's it.


## API overview

### `mobiedantic.Project`

The `Project` class manages MoBIE project.json files and orchestrates multiple datasets.

#### Initialization
```python
from pathlib import Path
from mobiedantic import Project

project = Project(path=Path('/path/to/project'))
```

#### Core Methods

- `initialize_model(description)`: Create a new project model with a description and spec version
  - `description`: Text description of the project
- `load()`: Load project.json from disk into the model
- `save(create_directory=True)`: Save the project model to project.json
- `new_dataset(name, make_default=False, overwrite=True)`: Create a new dataset within the project, returning a Dataset instance
  - `name`: Name of the new dataset
  - `make_default`: Whether to set this as the default dataset (auto-set to True for first dataset)
  - `overwrite`: Whether to overwrite existing directory if present (default: True)

### `mobiedantic.Dataset`

The `Dataset` class manages MoBIE dataset.json files and their associated data structures.

### Initialization
```python
from pathlib import Path
from mobiedantic import Dataset

dataset = Dataset(path=Path('/path/to/dataset'))
```

### Core Methods

- `load()`: Load dataset.json from disk into the model
- `save(create_directory=True)`: Save the dataset model to dataset.json
- `set_model(model)`: Set the pydantic DatasetSchema model directly
- `initialize_with_paths(path_dict, is2d, channel_index=0, data_format='ome.zarr')`: Initialize a new dataset with image sources from file paths
  - `path_dict`: Dictionary mapping source names to file paths
  - `is2d`: Boolean indicating if dataset is 2D or 3D
  - `channel_index`: Channel index for multi-channel images (default: 0)
  - `data_format`: Format of image data, e.g., 'ome.zarr', 'tiff' (default: 'ome.zarr')
- `add_sources(path_dict, channel_index=0, data_format='ome.zarr')`: Add additional image sources to an existing dataset
- `add_merged_grid(name, sources, positions=None, view_name='default')`: Create a merged grid view combining multiple image sources
  - `name`: Name for the merged grid source
  - `sources`: List of source names to merge
  - `positions`: Optional list of (x, y) grid positions for each source
  - `view_name`: Name of the view to add the merged grid to (default: 'default')
- `add_region_view(name, map_of_sources, view_name='default')`: Add a region-based view with associated segmentation tables
  - `name`: Name for the region source
  - `map_of_sources`: Dictionary mapping display names to lists of source names
  - `view_name`: Name of the view to add the region to (default: 'default')


## Project structure

### Directory Layout

```
mobiedantic/
├── src/mobiedantic/          # Main package source code
│   ├── __init__.py          # Exports Dataset and Project classes
│   └── generated.py         # Pydantic models (generated from JSON schemas)
├── tests/                    # Test suite
│   ├── test_api.py          # Tests for Dataset and Project APIs
│   └── test_schemas.py      # Tests for pydantic schema models
├── docs/                     # Documentation
│   └── PLAN.md              # This file
├── pyproject.toml           # Project metadata and dependencies
├── pixi.lock                # Committed lock file
├── LICENSE.txt              # BSD-3-Clause
└── README.md                # Package readme
```

### Dependencies

- **pydantic**: Core validation and data modeling library
- **Python ≥3.8**: Supports Python 3.8 through 3.13

