from pathlib import Path

import pytest

from mobiedantic import Dataset, Project


def test_project(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Test project')
    project.new_dataset('Dataset_1')
    project.save()
    assert (tmp_path / 'project.json').exists()
    assert len(project.model.datasets) == 1


def test_dataset(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Testing datasets')
    dataset1: Dataset = project.new_dataset('Dataset_1')
    sources1 = {
        'A01': Path('../../non-existent.zarr/A/01/0'),
        'A02': Path('../../non-existent.zarr/A/02/0'),
    }
    dataset1.initialize_with_paths(
        path_dict=sources1,
        is2d=True,
        channel_index=0,
    )
    dataset1.add_merged_grid(
        name='Merged_grid_view',
        sources=list(sources1),
    )
    assert len(dataset1.model.views['default'].sourceDisplays) == 1
    assert (
        dataset1.model.views['default'].sourceDisplays[0].imageDisplay.name.root
        == 'Merged_grid_view'
    )
    assert len(dataset1.model.views['default'].sourceTransforms) == 1
    assert (
        dataset1.model.views['default']
        .sourceTransforms[0]
        .mergedGrid.mergedGridSourceName.root
        == 'Merged_grid_view'
    )
    assert (
        len(dataset1.model.views['default'].sourceTransforms[0].mergedGrid.sources) == 2
    )
    sources2 = {
        'C01': Path('../../non-existent.zarr/C/01/0'),
        'C02': Path('../../non-existent.zarr/C/02/0'),
    }
    dataset1.add_sources(
        path_dict=sources2,
        channel_index=0,
        data_format='ome.zarr',
    )
    dataset1.add_merged_grid(
        name='Merged_grid_view_2',
        sources=list(sources2),
    )
    dataset1.add_region_view(
        name='wells',
        map_of_sources={
            'A01': ['A01_ch1', 'A01_ch2'],
            'B01': ['B01_ch1', 'B01_ch2'],
        },
    )
    assert len(dataset1.model.sources) == 4
    assert len(dataset1.model.views['default'].sourceDisplays) == 3
    assert (
        dataset1.model.views['default'].sourceDisplays[1].imageDisplay.name.root
        == 'Merged_grid_view_2'
    )
    assert len(dataset1.model.views['default'].sourceTransforms) == 2
    assert (
        dataset1.model.views['default']
        .sourceTransforms[1]
        .mergedGrid.mergedGridSourceName.root
        == 'Merged_grid_view_2'
    )
    assert (
        len(dataset1.model.views['default'].sourceTransforms[1].mergedGrid.sources) == 2
    )
    project.new_dataset('Dataset_2')
    assert len(project.model.datasets) == 2


def test_save_and_load(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Testing saving and loading')
    dataset_name = 'dataset1'
    dataset = project.new_dataset(dataset_name)
    sources = {
        'A01': '/path/to/source',
    }
    dataset.initialize_with_paths(
        path_dict=sources,
        is2d=True,
    )
    dataset.save()
    project.save()

    project_loaded = Project(tmp_path)
    project_loaded.load()

    assert project.model == project_loaded.model

    dataset_loaded = Dataset(tmp_path / dataset_name)
    dataset_loaded.load()

    assert dataset.model == dataset_loaded.model


def test_dataset_errors(tmp_path):
    filename = 'dataset.json'
    with open(tmp_path / filename, 'w'):
        pass
    with pytest.raises(ValueError, match=r"'path' needs to point to a directory"):
        Dataset(path=(tmp_path / filename))
    dataset_dir = tmp_path / 'dataset'
    dataset = Dataset(path=dataset_dir)
    with pytest.raises(ValueError, match=r'Dataset not initialized.'):
        dataset.save()
    sources = {
        'A01': '/path/to/source',
    }
    dataset.initialize_with_paths(
        path_dict=sources,
        is2d=True,
    )
    with pytest.raises(
        ValueError, match=r"Dataset folder doesn't exist yet and may not be created."
    ):
        dataset.save(create_directory=False)
    with pytest.raises(ValueError, match=r'Dataset file not found'):
        dataset.load()


def test_project_errors(tmp_path):
    project = Project(tmp_path / 'non-existent_subfolder')
    with pytest.raises(ValueError, match=r'Project not initialized'):
        project.save()
    with pytest.raises(ValueError, match=r'Project file not found'):
        project.load()
    with pytest.raises(ValueError, match=r'Project not initialized'):
        project.new_dataset('dataset1')
    project.initialize_model(description='Test raising errors.')
    with pytest.raises(
        ValueError, match=r"Project folder doesn't exist yet and may not be created."
    ):
        project.save(create_directory=False)
