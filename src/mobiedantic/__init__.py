from __future__ import annotations

import json
from pathlib import Path

from mobiedantic.generated import Dataset as DatasetSchema
from mobiedantic.generated import (
    ImageDisplay,
    ImageDisplay1,
    MergedGrid,
    Name,
    Source,
    RegionDisplay,
    RegionDisplay1,
)
from mobiedantic.generated import Project as ProjectSchema


class Dataset:
    path: Path
    model: DatasetSchema = None

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_dir():
            message = "'path' needs to point to a directory."
            raise ValueError(message)

    def save(self, *, create_directory: bool = True):
        if self.model is None:
            message = 'Dataset not initialized.'
            raise ValueError(message)
        if not self.path.exists() and not create_directory:
            message = "Dataset folder doesn't exist yet and may not be created."
            raise ValueError(message)
        self.path.mkdir(exist_ok=True)
        with open(self.path / 'dataset.json', 'w') as dataset_file:
            dataset_file.write(
                json.dumps(
                    self.model.model_dump(exclude_none=True, by_alias=True), indent=2
                )
            )

    def load(self):
        dataset_path = self.path / 'dataset.json'
        if not dataset_path.exists():
            message = f'Dataset file not found: {dataset_path}'
            raise ValueError(message)
        with open(dataset_path) as dataset_file:
            data = json.loads(dataset_file.read())
            self.model = DatasetSchema(**data)

    def set_model(self, model: DatasetSchema):
        self.model = model

    def initialize_with_paths(
        self,
        path_dict: dict[str, Path],
        *,
        is2d: bool,
        channel_index: int = 0,
        data_format: str = 'ome.zarr',
    ) -> None:
        sources = {}
        self._update_sources(
            sources=sources,
            path_dict=path_dict,
            channel_index=channel_index,
            data_format=data_format,
        )
        views_dict = {'default': {'uiSelectionGroup': 'view', 'isExclusive': True}}
        self.model = DatasetSchema(
            is2D=is2d,
            sources=sources,
            views=views_dict,
        )

    def _update_sources(
        self,
        sources: dict[str, Source],
        path_dict: dict[str, Path],
        channel_index: int = 0,
        data_format: str = 'ome.zarr',
    ):
        for name in path_dict:
            try:
                source_path = {
                    'channel': channel_index,
                    'relativePath': str(
                        Path(path_dict[name]).relative_to(self.path, walk_up=True)
                    ),
                }
            except (ValueError, TypeError):
                source_path = {
                    'channel': channel_index,
                    'absolutePath': str(Path(path_dict[name]).absolute()),
                }
            data = {
                'image': {
                    'imageData': {
                        data_format: source_path,
                    }
                }
            }
            sources[name] = Source(**data)

    def add_sources(
        self,
        path_dict: dict[str, Path],
        *,
        channel_index: int = 0,
        data_format: str = 'ome.zarr',
    ):
        self._update_sources(
            sources=self.model.sources,
            path_dict=path_dict,
            channel_index=channel_index,
            data_format=data_format,
        )

    def add_merged_grid(
        self,
        name: str,
        sources: list[str],
        positions: list[tuple[int, int]] | None = None,
        *,
        view_name: str = 'default',
    ) -> None:
        if self.model.views[view_name].sourceTransforms is None:
            self.model.views[view_name].sourceTransforms = []
        self.model.views[view_name].sourceTransforms.append(
            MergedGrid(
                mergedGrid={
                    'sources': sources,
                    'positions': positions,
                    'mergedGridSourceName': name,
                }
            )
        )
        if self.model.views[view_name].sourceDisplays is None:
            self.model.views[view_name].sourceDisplays = []
        self.model.views[view_name].sourceDisplays.append(
            ImageDisplay(
                imageDisplay=ImageDisplay1(
                    name=name,
                    color='white',
                    opacity=1.0,
                    contrastLimits=[0, 255],  # TODO: get from histograms/data
                    sources=[name],
                )
            )
        )

    def add_region_view(
        self,
        name: str,
        map_of_sources: dict[str, list[str]],
        *,
        view_name: str = 'default',
    ) -> None:
        # create table with single column 'region_id' containing source_names, and save as .tsv inside "tables/<name>" inside dataset folder
        table = 'region_id\t\n'
        for source_name in map_of_sources:
            table += f'{source_name}\t\n'
        tables_folder = self.path / 'tables' / name
        tables_folder.mkdir(parents=True, exist_ok=True)
        table_path = tables_folder / 'default.tsv'
        with open(table_path, 'w') as table_file:
            table_file.write(table)
        self.model.views[view_name].sourceDisplays.append(
            RegionDisplay(
                regionDisplay=RegionDisplay1(
                    name=name,
                    sources=map_of_sources,
                    tableSource=name,
                    lut='glasbey',
                    opacity=0.5,
                )
            )
        )


class Project:
    path: Path
    model: ProjectSchema = None

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialize_model(self, description: str) -> None:
        self.model = ProjectSchema(
            datasets=[], defaultDataset='', description=description, specVersion='0.3.0'
        )

    def new_dataset(
        self, name: str, *, make_default: bool = False, overwrite: bool = True
    ) -> Dataset:
        if self.model is None:
            message = 'Project not initialized.'
            raise ValueError(message)
        if len(self.model.datasets) == 0:
            make_default = True
        dataset_folder = self.path / name
        dataset_folder.mkdir(exist_ok=overwrite, parents=True)
        self.model.datasets.append(Name(name))
        if make_default:
            self.model.defaultDataset = name
        return Dataset(path=dataset_folder)

    def load(self):
        project_path = self.path / 'project.json'
        if not project_path.exists():
            message = f'Project file not found: {project_path}'
            raise ValueError(message)
        with open(project_path) as project_file:
            data = json.loads(project_file.read())
            self.model = ProjectSchema(**data)

    def save(self, *, create_directory: bool = True):
        if self.model is None:
            message = 'Project not initialized.'
            raise ValueError(message)
        if not self.path.exists() and not create_directory:
            message = "Project folder doesn't exist yet and may not be created."
            raise ValueError(message)
        self.path.mkdir(exist_ok=True)
        with open(self.path / 'project.json', 'w') as project_file:
            project_file.write(
                json.dumps(
                    self.model.model_dump(exclude_none=True, by_alias=True), indent=2
                )
            )

    # @classmethod
    # def create(cls, path: Path) -> "Project":
