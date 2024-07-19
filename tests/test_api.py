from pathlib import Path

from mobiedantic import Project


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
    dataset1 = project.new_dataset('Dataset_1')
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

    dataset2 = project.new_dataset('Dataset_2')
    sources2 = {
        'C01': Path('../../non-existent.zarr/C/01/0'),
        'C02': Path('../../non-existent.zarr/C/02/0'),
    }
    dataset2.initialize_with_paths(
        path_dict=sources2,
        is2d=False,
        channel_index=0,
    )
