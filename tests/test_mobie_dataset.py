from mobiedantic.generated import MoBIEDatasetSchema, Schema


def test_dataset_schema():
    dataset = MoBIEDatasetSchema(
        is2D=True,
        sources={
            'A01_C3': {'image': {'imageData': {'ome.zarr': {'relativePath': '.'}}}}
        },
        views={'default': {'uiSelectionGroup': 'any', 'isExclusive': True}},
    )

    dataset.sources['A01_C1'] = Schema(
        {'image': {'imageData': {'ome.zarr': {'relativePath': '.'}}}}
    )

    assert len(dataset.sources) == 2
