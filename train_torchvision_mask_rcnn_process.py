from ikomia import dataprocess, utils
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain, datasetio
import os
import copy
from train_torchvision_mask_rcnn import trainer


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class TrainMaskRcnnParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_name"] = 'MaskRCNN'
        self.cfg["batch_size"] = 8
        self.cfg["classes"] = 2
        self.cfg["epochs"] = 15
        self.cfg["num_workers"] = 0
        self.cfg["input_size"] = 224
        self.cfg["learning_rate"] = 0.005
        self.cfg["momentum"] = 0.9
        self.cfg["weight_decay"] = 0.0005
        self.cfg["export_pth"] = True
        self.cfg["export_onnx"] = False
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/models/"

    def set_values(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["classes"] = int(param_map["classes"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["num_workers"] = int(param_map["num_workers"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["learning_rate"] = float(param_map["learning_rate"])
        self.cfg["momentum"] = float(param_map["momentum"])
        self.cfg["weight_decay"] = float(param_map["weight_decay"])
        self.cfg["export_pth"] = utils.strtobool(param_map["export_pth"])
        self.cfg["export_onnx"] = utils.strtobool(param_map["export_onnx"])
        self.cfg["output_folder"] = param_map["output_folder"]


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class TrainMaskRcnn(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        if param is None:
            self.set_param_object(TrainMaskRcnnParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.trainer = trainer.MaskRCNN(self.get_param_object())
        self.enable_tensorboard(False)

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process

        # Get parameters :
        param = self.get_param_object()

        # Get dataset path from input
        dataset_input = self.get_input(0)
        param.cfg["classes"] = dataset_input.get_category_count()

        # Call begin_task_run for initialization
        self.begin_task_run()

        print("Starting training job...")
        self.trainer.launch(dataset_input, self.on_epoch_end)

        print("Training job finished.")

        # Call end_task_run to finalize process
        self.end_task_run()

    def on_epoch_end(self, metrics):
        # Step progress bar:
        self.emit_step_progress()
        # Log metrics
        self.log_metrics(metrics)

    def stop(self):
        super().stop()
        self.trainer.stop()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class TrainMaskRcnnFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_torchvision_mask_rcnn"
        self.info.short_description = "Training process for Mask R-CNN convolutional network."
        self.info.description = "Training process for Mask R-CNN convolutional network. The process enables " \
                                "to train Mask R-CNN network with ResNet50 backbone for transfer learning. " \
                                "You must connect this process behind a suitable dataset loader (with segmentation " \
                                "masks). You can find one in the Ikomia marketplace or implement your own via " \
                                "the Ikomia API."
        self.info.authors = "Ikomia"
        self.info.version = "1.3.0"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.repo = "https://github.com/Ikomia-dev"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icons/pytorch-logo.png"
        self.info.keywords = "object,detection,instance,segmentation,ResNet,pytorch,train"

    def create(self, param=None):
        # Create process object
        return TrainMaskRcnn(self.info.name, param)
