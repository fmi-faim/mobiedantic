import json
from urllib.request import urlopen

import pytest
from mobie.validation import (
    validate_dataset,
)
from pydantic import ValidationError

from mobiedantic.generated import (
    Dataset,
    Source,
)


def test_dataset_schema(tmp_path):
    dataset = Dataset(
        is2D=True,
        sources={
            'A01_C3': {'image': {'imageData': {'ome.zarr': {'relativePath': '.'}}}}
        },
        views={
            'default': {
                'uiSelectionGroup': 'any',
                'isExclusive': True,
                'sourceDisplays': [
                    {
                        'imageDisplay': {
                            'color': 'white',
                            'contrastLimits': [0, 255],
                            'opacity': 1,
                            'name': 'Image_Display_1',
                            'sources': ['A01_C3'],
                        }
                    },
                ],
            },
            'otherView': {'uiSelectionGroup': 'any', 'isExclusive': True},
        },
    )

    dataset.sources['A01_C1'] = Source(
        {'image': {'imageData': {'ome.zarr': {'relativePath': '.'}}}}
    )

    assert len(dataset.sources) == 2
    assert len(dataset.views) == 2
    with open(tmp_path / 'dataset.json', 'w') as dataset_file:
        dataset_file.write(
            json.dumps(dataset.model_dump(exclude_none=True, by_alias=True), indent=2)
        )
    validate_dataset(tmp_path, require_local_data=False)


def test_dataset_missing_default_view():
    with pytest.raises(ValidationError):
        Dataset(
            is2D=True,
            sources={
                'A01_C3': {'image': {'imageData': {'ome.zarr': {'relativePath': '.'}}}}
            },
            views={
                'otherView': {'uiSelectionGroup': 'any', 'isExclusive': True},
            },
        )


@pytest.mark.parametrize(
    'url',
    [
        'https://raw.githubusercontent.com/mobie/clem-example-project/main/data/hela/dataset.json',
        'https://raw.githubusercontent.com/mobie/clem-example-project/main/data/yeast/dataset.json',
        # NB: extra centerAtOrigin and encodeSource in mergedGrid
        'https://raw.githubusercontent.com/mobie/covid-if-project/main/data/20200406_164555_328/dataset.json',
        # NB: extra timepoints in dataset
        # 'https://raw.githubusercontent.com/mobie/arabidopsis-root-lm-project/main/data/arabidopsis-root/dataset.json',
        # NB: spotDisplay is lacking sources (?)
        # 'https://raw.githubusercontent.com/mobie/mouse-embryo-spatial-transcriptomics-project/main/data/embryo3/dataset.json',
    ],
)
def test_existing_datasets(url):
    data = json.loads(urlopen(url).read())

    # ensure validation with jsonschema
    # NB: this can take quite long with some datasets
    # validate_with_schema(data, "dataset")

    dataset = Dataset(**data)
    assert dataset.model_dump(by_alias=True, exclude_none=True) == data

    # NB: for debugging, we can dig into nested field in detail
    # for view in dataset.views:
    #     for idx, source_display in enumerate(dataset.views[view].sourceDisplays):
    #         if isinstance(source_display, ImageDisplay):
    #             assert source_display.imageDisplay.model_dump(exclude_none=True, by_alias=True) == data["views"][view]["sourceDisplays"][idx]["imageDisplay"]
    #         if isinstance(source_display, SegmentationDisplay):
    #             assert source_display.segmentationDisplay.model_dump(exclude_none=True, by_alias=True) == data["views"][view]["sourceDisplays"][idx]["segmentationDisplay"]
    #         if isinstance(source_display, SpotDisplay):
    #             assert source_display.spotDisplay.model_dump(exclude_none=True, by_alias=True) == data["views"][view]["sourceDisplays"][idx]["spotDisplay"]
    #         if isinstance(source_display, RegionDisplay):
    #             assert source_display.regionDisplay.model_dump(exclude_none=True, by_alias=True) == data["views"][view]["sourceDisplays"][idx]["regionDisplay"]
    # for source in dataset.sources:
    #     if isinstance(source, MoBIESourceSchema):
    #             assert source.image.model_dump(exclude_none=True, by_alias=True) == data["sources"][source]["image"]
    #     if isinstance(source, MoBIESourceSchema1):
    #             assert source.segmentation.model_dump(exclude_none=True, by_alias=True) == data["sources"][source]["segmentation"]
    #     if isinstance(source, MoBIESourceSchema2):
    #             assert source.spots.model_dump(exclude_none=True, by_alias=True) == data["sources"][source]["spots"]
    #     if isinstance(source, MoBIESourceSchema3):
    #             assert source.regions.model_dump(exclude_none=True, by_alias=True) == data["sources"][source]["regions"]
