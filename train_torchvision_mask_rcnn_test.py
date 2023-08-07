import logging
from ikomia.utils.tests import run_for_test

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train torchvision mask rcnn =====")
    input_dataset = t.get_input(0)
    params = t.get_parameters()
    params["epochs"] = "2"
    params["batch_size"] = "1"
    t.set_parameters(params)
    input_dataset.load(data_dict["datasets"]["instance_segmentation"]["dataset_coco"])
    yield run_for_test(t)