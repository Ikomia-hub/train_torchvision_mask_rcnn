from ikomia import dataprocess
from ikomia.dnn import dnntrain, datasetio
import os
import copy
# Your imports below
import MaskRCNN


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class MaskRCNNTrainParam(dataprocess.CDnnTrainProcessParam):

    def __init__(self):
        dataprocess.CDnnTrainProcessParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'MaskRCNN'
        self.batch_size = 8
        self.classes = 2
        self.epochs = 15
        self.num_workers = 0
        self.input_size = 224
        self.learning_rate = 0.005
        self.weight_decay = 0.0005
        self.export_pth = True
        self.export_onnx = False
        self.output_folder = os.path.dirname(os.path.realpath(__file__)) + "/models/"

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        super().setParamMap(paramMap)
        self.num_workers = int(paramMap["num_workers"])
        self.input_size = int(paramMap["input_size"])
        self.export_pth = bool(paramMap["export_pth"])
        self.export_onnx = bool(paramMap["export_onnx"])
        self.output_folder = paramMap["output_folder"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = super().getParamMap()
        param_map["num_workers"] = str(self.num_workers)
        param_map["input_size"] = str(self.input_size)
        param_map["export_pth"] = str(self.export_pth)
        param_map["export_onnx"] = str(self.export_onnx)
        param_map["output_folder"] = self.output_folder
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class MaskRCNNTrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        self.addInput(datasetio.IkDatasetIO())

        # Create parameters class
        if param is None:
            self.setParam(MaskRCNNTrainParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.trainer = MaskRCNN.MaskRCNN(self.getParam())

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.epochs
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Get dataset path from input
        dataset_input = self.getInput(0)
        param.classes = dataset_input.getCategoryCount()

        print("Starting training job...")
        self.trainer.launch(dataset_input, self.on_epoch_end)

        print("Training job finished.")

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def on_epoch_end(self, metrics):
        # Step progress bar:
        self.emitStepProgress()
        # Log metrics
        self.log_metrics(metrics)

    def stop(self):
        super().stop()
        self.trainer.stop()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MaskRCNNTrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "MaskRCNN Train"
        self.info.shortDescription = "Training process for Mask R-CNN convolutional network."
        self.info.description = "Training process for Mask R-CNN convolutional network. The process enables " \
                                "to train Mask R-CNN network with ResNet50 backbone for transfer learning. " \
                                "You must connect this process behind a suitable dataset loader (with segmentation " \
                                "masks). You can find one in the Ikomia marketplace or implement your own via " \
                                "the Ikomia API."
        self.info.authors = "Ikomia"
        self.info.version = "1.1.2"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.repo = "https://github.com/Ikomia-dev"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Train"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.keywords = "object,detection,instance,segmentation,ResNet,pytorch,train"

    def create(self, param=None):
        # Create process object
        return MaskRCNNTrainProcess(self.info.name, param)
