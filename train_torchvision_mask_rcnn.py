from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from train_torchvision_mask_rcnn.train_torchvision_mask_rcnn_process import TrainMaskRcnnFactory
        # Instantiate process object
        return TrainMaskRcnnFactory()

    def getWidgetFactory(self):
        from train_torchvision_mask_rcnn.train_torchvision_mask_rcnn_widget import TrainMaskRcnnWidgetFactory
        # Instantiate associated widget object
        return TrainMaskRcnnWidgetFactory()
