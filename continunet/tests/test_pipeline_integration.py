"""End to end integration test for ContinUNet source finding pipeline using SDC1
256x256 FITS image."""

from continunet.image.fits import ImageSquare
from continunet.image.processing import PreProcessor, PostProcessor
from continunet.network.unet import Unet
from continunet.constants import TRAINED_MODEL


def test_example_fits_pipeline_runs(example_fits_path):
    """Run the full pipeline on a small FITS file to ensure it completes."""
    image = ImageSquare(example_fits_path)

    # Preprocessing
    pre = PreProcessor(image)
    data = pre.process()
    assert data is not None
    assert data.shape[-1] == 1  # should be single-channel

    # UNet inference (mocked weights if necessary)
    model = Unet(data.shape[1:], image=data, trained_model=TRAINED_MODEL)
    decoded = model.decode_image()
    assert decoded.shape == data.shape

    # Postprocessing
    post = PostProcessor(
        reconstructed_image=decoded,
        pre_processed_image=pre,
        threshold="default",
        sigma_snr=None,
        rms_box="default",
        clean_maps=False,
    )
    seg_map = post.get_segmentation_map()
    assert seg_map is not None
    assert seg_map.shape[:2] == data.shape[1:3]
