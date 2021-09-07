from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MaskRCNNTrain(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from MaskRCNNTrain.MaskRCNNTrain_process import MaskRCNNTrainProcessFactory
        # Instantiate process object
        return MaskRCNNTrainProcessFactory()

    def getWidgetFactory(self):
        from MaskRCNNTrain.MaskRCNNTrain_widget import MaskRCNNTrainWidgetFactory
        # Instantiate associated widget object
        return MaskRCNNTrainWidgetFactory()
