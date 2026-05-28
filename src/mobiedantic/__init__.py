from __future__ import annotations

import csv
import json
import os
from loguru import logger
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
    SegmentationDisplay,
    SegmentationDisplay1,
    ValueLimits,
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
            self.set_model(DatasetSchema(**data))

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

    def _as_source_path(self, path: Path, *, channel_index: int | None = None):
        source_path = {}
        if channel_index is not None:
            source_path['channel'] = channel_index
        resolved_path = Path(path)
        logger.info(f'{resolved_path=}, {self.path=}')
        try:
            source_path['relativePath'] = str(
                resolved_path.relative_to(self.path).as_posix()
            )  # TODO update walk_up=True when pinning python 3.12+
        except ValueError:
            try:
                source_path['relativePath'] = os.path.relpath(
                    resolved_path, self.path
                ).replace(os.sep, '/')
            except ValueError:
                source_path['absolutePath'] = str(resolved_path.absolute())
        except TypeError:
            source_path['absolutePath'] = str(Path(path).absolute())
        logger.info(f'{source_path=}')
        return source_path

    def _update_sources(
        self,
        sources: dict[str, Source],
        path_dict: dict[str, Path],
        channel_index: int = 0,
        data_format: str = 'ome.zarr',
    ):
        for name in path_dict:
            source_path = self._as_source_path(
                path_dict[name], channel_index=channel_index
            )
            data = {
                'image': {
                    'imageData': {
                        data_format: source_path,
                    }
                }
            }
            sources[name] = Source(**data)

    def add_image_sources(
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

    def add_segmentation_sources(
        self,
        path_dict: dict[str, Path],
        *,
        table_path_dict: dict[str, Path] | None = None,
        channel_index: int = 0,
        data_format: str = 'ome.zarr',
    ) -> None:
        if table_path_dict is not None:
            unknown_sources = set(table_path_dict) - set(path_dict)
            if unknown_sources:
                message = (
                    'table_path_dict has keys that are not present in path_dict: '
                    f'{sorted(unknown_sources)}'
                )
                raise ValueError(message)

        for name, path in path_dict.items():
            data = {
                'segmentation': {
                    'imageData': {
                        data_format: self._as_source_path(
                            path, channel_index=channel_index
                        ),
                    }
                }
            }
            if table_path_dict is not None and name in table_path_dict:
                data['segmentation']['tableData'] = {
                    'tsv': self._as_source_path(table_path_dict[name])
                }
            self.model.sources[name] = Source(**data)

    def _resolve_table_file_and_root(self, table_path: Path) -> tuple[Path, Path]:
        direct_path = Path(table_path)
        dataset_relative_path = self.path / direct_path

        for candidate in (direct_path, dataset_relative_path):
            if candidate.is_dir():
                default_table = candidate / 'default.tsv'
                if default_table.exists():
                    return default_table, candidate
                message = f'Default table {default_table} does not exist.'
                raise ValueError(message)

            if candidate.is_file():
                return candidate, candidate.parent

        message = f'Spot table file not found: {table_path}'
        raise ValueError(message)

    def _compute_spot_bounding_box(
        self, table_file_path: Path
    ) -> tuple[list[float], list[float]]:
        delimiter = '\t' if table_file_path.suffix.lower() == '.tsv' else ','

        with open(table_file_path, newline='') as table_file:
            reader = csv.DictReader(table_file, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            required_columns = {'x', 'y'}
            missing_columns = required_columns - set(fieldnames)
            if missing_columns:
                message = (
                    f'Missing required columns {sorted(missing_columns)} in spot table: '
                    f'{table_file_path}'
                )
                raise ValueError(message)

            has_z = 'z' in fieldnames
            min_values: list[float] | None = None
            max_values: list[float] | None = None
            row_count = 0

            for row in reader:
                coordinate_names = ['x', 'y', 'z'] if has_z else ['x', 'y']
                try:
                    values = [float(row[name]) for name in coordinate_names]
                except (TypeError, ValueError):
                    message = (
                        f'Invalid numeric coordinate values in spot table: '
                        f'{table_file_path}'
                    )
                    raise ValueError(message) from None

                if min_values is None:
                    min_values = values.copy()
                    max_values = values.copy()
                else:
                    min_values = [
                        min(current, value)
                        for current, value in zip(min_values, values)
                    ]
                    max_values = [
                        max(current, value)
                        for current, value in zip(max_values, values)
                    ]
                row_count += 1

            if row_count == 0 or min_values is None or max_values is None:
                message = f'Spot table is empty: {table_file_path}'
                raise ValueError(message)

            return min_values, max_values

    def add_spots_sources(
        self,
        table_path_dict: dict[str, Path],
        *,
        unit: str = 'micrometer',
    ) -> None:
        for name, table_path in table_path_dict.items():
            table_file_path, table_root_path = self._resolve_table_file_and_root(
                table_path
            )
            bounding_box_min, bounding_box_max = self._compute_spot_bounding_box(
                table_file_path
            )
            self.model.sources[name] = Source(
                **{
                    'spots': {
                        'boundingBoxMin': bounding_box_min,
                        'boundingBoxMax': bounding_box_max,
                        'tableData': {'tsv': self._as_source_path(table_root_path)},
                        'unit': unit,
                    }
                }
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

    def add_image_display(
        self,
        name: str,
        sources: list[str],
        *,
        color: str = 'white',
        contrast_limits: list[float] = [0, 255],
        opacity: float = 1.0,
        visible: bool | None = None,
        view_name: str = 'default',
    ) -> None:
        if self.model.views[view_name].sourceDisplays is None:
            self.model.views[view_name].sourceDisplays = []
        self.model.views[view_name].sourceDisplays.append(
            ImageDisplay(
                imageDisplay=ImageDisplay1(
                    name=name,
                    color=color,
                    opacity=opacity,
                    contrastLimits=contrast_limits,
                    sources=sources,
                    visible=visible,
                )
            )
        )

    def add_region_display(
        self,
        name: str,
        map_of_sources: dict[str, list[str]],
        *,
        visible: bool | None = None,
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
        # add table source to dataset
        self.model.sources[name] = Source(
            {'regions': {'tableData': {'tsv': {'relativePath': f'tables/{name}'}}}}
        )
        if self.model.views[view_name].sourceDisplays is None:
            self.model.views[view_name].sourceDisplays = []
        self.model.views[view_name].sourceDisplays.append(
            RegionDisplay(
                regionDisplay=RegionDisplay1(
                    name=name,
                    sources=map_of_sources,
                    tableSource=name,
                    lut='glasbey',
                    opacity=0.5,
                    visible=visible,
                )
            )
        )

    def add_segmentation_display(
        self,
        name: str,
        sources: list[str],
        *,
        opacity: float = 0.5,
        lut: str = 'glasbey',
        visible: bool | None = None,
        view_name: str = 'default',
    ) -> None:
        if self.model.views[view_name].sourceDisplays is None:
            self.model.views[view_name].sourceDisplays = []
        self.model.views[view_name].sourceDisplays.append(
            SegmentationDisplay(
                segmentationDisplay=SegmentationDisplay1(
                    name=name,
                    sources=sources,
                    opacity=opacity,
                    lut=lut,
                    valueLimits=ValueLimits([0, 255]),
                    visible=visible,
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
