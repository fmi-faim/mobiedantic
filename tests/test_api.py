import csv
from pathlib import Path

import pytest
from mobie.validation import validate_dataset

from mobiedantic import Dataset, Project


def _validate_dataset_on_disk(dataset: Dataset) -> None:
    dataset.save()
    with pytest.warns(
        DeprecationWarning,
        match=r'Automatically retrieving remote references can be a security vulnerability and is discouraged by the JSON Schema specifications.',
    ):
        validate_dataset(dataset.path, require_local_data=False)


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
        'A01_ch1': Path('../../non-existent.zarr/A/01/0'),
        'B01_ch1': Path('../../non-existent.zarr/B/01/0'),
    }
    dataset1.initialize_with_paths(
        path_dict=sources1,
        is2d=True,
        channel_index=0,
    )
    dataset1.add_merged_grid(
        name='ch1',
        sources=list(sources1),
    )
    dataset1.add_image_display(
        name='ch1',
        sources=['ch1'],
        color='white',
        contrast_limits=[0, 255],
        opacity=0.5,
    )
    assert len(dataset1.model.views['default'].sourceDisplays) == 1
    assert (
        dataset1.model.views['default'].sourceDisplays[0].imageDisplay.name.root
        == 'ch1'
    )
    assert len(dataset1.model.views['default'].sourceTransforms) == 1
    assert (
        dataset1.model.views['default']
        .sourceTransforms[0]
        .mergedGrid.mergedGridSourceName.root
        == 'ch1'
    )
    assert (
        len(dataset1.model.views['default'].sourceTransforms[0].mergedGrid.sources) == 2
    )
    sources2 = {
        'A01_ch2': Path('../../non-existent.zarr/A/01/0'),
        'B01_ch2': Path('../../non-existent.zarr/B/01/0'),
    }
    dataset1.add_image_sources(
        path_dict=sources2,
        channel_index=1,
        data_format='ome.zarr',
    )
    dataset1.add_merged_grid(
        name='ch2',
        sources=list(sources2),
    )
    dataset1.add_image_display(
        name='ch2',
        sources=['ch2'],
        color='red',
        contrast_limits=[0, 255],
        opacity=0.5,
    )
    dataset1.add_region_view(
        name='wells',
        map_of_sources={
            'A01': ['A01_ch1', 'A01_ch2'],
            'B01': ['B01_ch1', 'B01_ch2'],
        },
    )
    assert len(dataset1.model.sources) == 5
    assert len(dataset1.model.views['default'].sourceDisplays) == 3
    assert (
        dataset1.model.views['default'].sourceDisplays[1].imageDisplay.name.root
        == 'ch2'
    )
    assert len(dataset1.model.views['default'].sourceTransforms) == 2
    assert (
        dataset1.model.views['default']
        .sourceTransforms[1]
        .mergedGrid.mergedGridSourceName.root
        == 'ch2'
    )
    assert (
        len(dataset1.model.views['default'].sourceTransforms[1].mergedGrid.sources) == 2
    )
    _validate_dataset_on_disk(dataset1)
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


def test_add_segmentation_sources(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Testing segmentation sources')
    dataset: Dataset = project.new_dataset('Dataset_1')
    dataset.initialize_with_paths(path_dict={'A01': '/path/to/source'}, is2d=True)

    segmentations = {'S01': Path('/path/to/segmentation')}
    segmentation_table_root = dataset.path / 'tables' / 'segmentation_table'
    segmentation_table_root.mkdir(parents=True)
    with open(segmentation_table_root / 'default.tsv', 'w', newline='') as table_file:
        writer = csv.writer(table_file, delimiter='\t')
        writer.writerow(['label_id', 'anchor_x', 'anchor_y', 'name'])
        writer.writerow([1, 0.0, 0.0, 'segment_1'])
    segmentation_tables = {'S01': segmentation_table_root}
    dataset.add_segmentation_sources(
        path_dict=segmentations,
        table_path_dict=segmentation_tables,
        channel_index=1,
        data_format='ome.zarr',
    )

    segmentation_data = dataset.model.sources['S01'].root.segmentation
    assert segmentation_data.imageData.ome_zarr is not None
    assert segmentation_data.imageData.ome_zarr.channel == 1
    assert segmentation_data.tableData is not None

    dataset.add_segmentation_sources(path_dict={'S02': Path('/path/to/seg2')})
    assert dataset.model.sources['S02'].root.segmentation.tableData is None

    with pytest.raises(ValueError, match=r'table_path_dict has keys'):
        dataset.add_segmentation_sources(
            path_dict={'S03': Path('/path/to/seg3')},
            table_path_dict={'UNKNOWN': Path('/path/to/table')},
        )
    _validate_dataset_on_disk(dataset)


def test_add_spots_sources(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Testing spot sources')
    dataset: Dataset = project.new_dataset('Dataset_1')
    dataset.initialize_with_paths(path_dict={'A01': '/path/to/source'}, is2d=True)

    spots_table_root = dataset.path / 'tables' / 'spots_1'
    spots_table_root.mkdir(parents=True)
    spots_table = spots_table_root / 'default.tsv'
    with open(spots_table, 'w', newline='') as table_file:
        writer = csv.writer(table_file, delimiter='\t')
        writer.writerow(['spot_id', 'x', 'y', 'z'])
        writer.writerow([1, 3.5, 2.0, 9.0])
        writer.writerow([2, -1.0, 5.5, 4.0])

    dataset.add_spots_sources({'spots_1': spots_table_root})

    spot_data = dataset.model.sources['spots_1'].root.spots
    assert spot_data.boundingBoxMin == [-1.0, 2.0, 4.0]
    assert spot_data.boundingBoxMax == [3.5, 5.5, 9.0]
    assert spot_data.unit == 'micrometer'
    assert spot_data.tableData.tsv.relativePath == 'tables/spots_1'
    _validate_dataset_on_disk(dataset)


def test_add_spots_sources_errors(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Testing spot source errors')
    dataset: Dataset = project.new_dataset('Dataset_1')
    dataset.initialize_with_paths(path_dict={'A01': '/path/to/source'}, is2d=True)

    missing_columns_table = tmp_path / 'missing_columns.tsv'
    with open(missing_columns_table, 'w', newline='') as table_file:
        writer = csv.writer(table_file, delimiter='\t')
        writer.writerow(['spot_id', 'x'])
        writer.writerow([1, 3.0])
    with pytest.raises(ValueError, match=r'Missing required columns'):
        dataset.add_spots_sources({'spots_missing_cols': missing_columns_table})

    invalid_values_table = tmp_path / 'invalid_values.tsv'
    with open(invalid_values_table, 'w', newline='') as table_file:
        writer = csv.writer(table_file, delimiter='\t')
        writer.writerow(['spot_id', 'x', 'y'])
        writer.writerow([1, 'not-a-number', 2.0])
    with pytest.raises(ValueError, match=r'Invalid numeric coordinate values'):
        dataset.add_spots_sources({'spots_invalid': invalid_values_table})

    empty_table = tmp_path / 'empty.tsv'
    with open(empty_table, 'w', newline='') as table_file:
        writer = csv.writer(table_file, delimiter='\t')
        writer.writerow(['spot_id', 'x', 'y'])
    with pytest.raises(ValueError, match=r'Spot table is empty'):
        dataset.add_spots_sources({'spots_empty': empty_table})


def test_add_segmentation_display(tmp_path):
    project = Project(tmp_path)
    project.initialize_model(description='Testing segmentation display')
    dataset: Dataset = project.new_dataset('Dataset_1')
    dataset.initialize_with_paths(path_dict={'A01': '/path/to/source'}, is2d=True)

    # Add segmentation sources
    segmentations = {
        'seg_01': Path('/path/to/segmentation1'),
        'seg_02': Path('/path/to/segmentation2'),
    }
    segmentation_table_root = dataset.path / 'tables' / 'segmentation_table'
    segmentation_table_root.mkdir(parents=True)
    with open(segmentation_table_root / 'default.tsv', 'w', newline='') as table_file:
        writer = csv.writer(table_file, delimiter='\t')
        writer.writerow(['label_id', 'anchor_x', 'anchor_y', 'name'])
        writer.writerow([1, 0.0, 0.0, 'segment_1'])
        writer.writerow([2, 1.0, 1.0, 'segment_2'])
    segmentation_tables = {
        'seg_01': segmentation_table_root,
        'seg_02': segmentation_table_root,
    }
    dataset.add_segmentation_sources(
        path_dict=segmentations,
        table_path_dict=segmentation_tables,
        channel_index=0,
        data_format='ome.zarr',
    )

    # Create merged grid from segmentation sources
    dataset.add_merged_grid(
        name='seg_merged',
        sources=list(segmentations),
    )

    # Add segmentation display pointing to merged grid
    dataset.add_segmentation_display(
        name='seg_display',
        sources=['seg_merged'],
    )

    # Verify display was added with defaults
    assert len(dataset.model.views['default'].sourceDisplays) == 1
    seg_display = dataset.model.views['default'].sourceDisplays[0]
    assert seg_display.segmentationDisplay.name.root == 'seg_display'
    assert [s.root for s in seg_display.segmentationDisplay.sources] == ['seg_merged']
    assert seg_display.segmentationDisplay.opacity.root == 0.5
    assert seg_display.segmentationDisplay.lut == 'glasbey'

    # Add another display with custom parameters
    dataset.add_segmentation_display(
        name='seg_display_2',
        sources=['seg_01', 'seg_02'],
        opacity=0.7,
        lut='viridis',
    )

    assert len(dataset.model.views['default'].sourceDisplays) == 2
    seg_display_2 = dataset.model.views['default'].sourceDisplays[1]
    assert seg_display_2.segmentationDisplay.name.root == 'seg_display_2'
    assert [s.root for s in seg_display_2.segmentationDisplay.sources] == ['seg_01', 'seg_02']
    assert seg_display_2.segmentationDisplay.opacity.root == 0.7
    assert seg_display_2.segmentationDisplay.lut == 'viridis'

    _validate_dataset_on_disk(dataset)


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
