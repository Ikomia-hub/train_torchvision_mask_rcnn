# MaskRCNNTrain

Ikomia plugin to train Mask RCNN instance segmentation model. This plugin is based on [PyTorch torchvision implementation](https://github.com/pytorch/vision).

### How to use it?
Here are the steps:

1. Create Ikomia account for free [here](https://ikomia.com/accounts/signup/) (if you don't have one)
2. Install [Ikomia software](https://ikomia.com/en/download)
3. Launch the software and log in with your credentials
4. Open Ikomia Store and install MaskRCNNTrain plugin
5. Install also a dataset loader plugin that fits to your data or implement it
6. Add the dataset loader to the workflow
7. Add the MaskRCNNTrain algorithm to the workflow
8. Start the workflow and evaluate the training thanks to [MLflow](https://www.mlflow.org/) integration

That's it!

**Note**: consult [this tutorial](https://blog.ikomia.com/2021/01/train-deep-learning-models-with-ikomia/) if you need more information.
