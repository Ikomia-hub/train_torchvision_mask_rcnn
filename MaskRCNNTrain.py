from ikomia import dataprocess
import MaskRCNNTrain_process as processMod
import MaskRCNNTrain_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MaskRCNNTrain(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.MaskRCNNTrainProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.MaskRCNNTrainWidgetFactory()
