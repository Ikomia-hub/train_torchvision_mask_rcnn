import os
import copy
import torch
from datetime import datetime
from ikomia.dnn.torch import models, datasetmapper
import ikomia.dnn.torch.utils as ikutils
from torchvisionref import transforms
from torchvisionref import utils
from torchvisionref.engine import train_one_epoch, evaluate


class MaskRCNN:
    def __init__(self, parameters):
        self.stop_train = False
        self.parameters = parameters
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _get_transforms(self, train):
        t = []
        if train:
            t.append(transforms.ToTensor())
            t.append(transforms.RandomHorizontalFlip(0.5))
        else:
            t.append(transforms.ToTensor())

        return transforms.Compose(t)

    def load_data(self, ik_dataset):
        print("Initializing Datasets and Dataloaders...")

        dataset = datasetmapper.TorchDatasetMapper(ik_dataset.data, ik_dataset.has_bckgnd_class, self._get_transforms(train=True))
        dataset_eval = datasetmapper.TorchDatasetMapper(ik_dataset.data, ik_dataset.has_bckgnd_class, self._get_transforms(train=False))

        # Split the dataset in train and test set
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        indices = torch.randperm(dataset_size).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:train_size])
        dataset_eval = torch.utils.data.Subset(dataset_eval, indices[train_size:])

        # Define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.parameters.batch_size, shuffle=True, 
                                                    num_workers=self.parameters.num_workers, 
                                                    collate_fn=utils.collate_fn)

        data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=1, shuffle=False, 
                                                        num_workers=self.parameters.num_workers,
                                                        collate_fn=utils.collate_fn)
        return data_loader, data_loader_eval

    def init_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer_ft = torch.optim.SGD(params, lr=self.parameters.learning_rate, momentum=self.parameters.momentum,
                                       weight_decay=self.parameters.weight_decay)
        return optimizer_ft

    def train_model(self, model, data_loader, data_loader_eval, optimizer, on_epoch_end):
        metrics = {}
        best_m_ap = -1

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(self.parameters.epochs):
            # Train for one epoch, printing every 10 iterations
            train_metrics = train_one_epoch(model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            eval_metrics = evaluate(model, data_loader_eval, device=self.device)

            m_ap = eval_metrics.coco_eval["bbox"].stats[0]
            if m_ap > best_m_ap:
                best_m_ap = m_ap
                best_model_wts = copy.deepcopy(model.state_dict())

            metrics["Loss"] = train_metrics.meters["loss"].global_avg
            metrics["mAP"] = m_ap
            metrics["Best mAP"] = best_m_ap
            on_epoch_end(metrics)

            if self.stop_train:
                break

        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def launch(self, dataset, on_epoch_end):
        self.stop_train = False

        # Load dataset
        data_loader, data_loader_test = self.load_data(dataset)

        # Initialize the model for this run
        class_count = self.parameters.classes
        if not dataset.has_bckgnd_class:
            class_count = class_count + 1

        model = models.mask_rcnn(train_mode=True,
                                   classes=class_count,
                                   input_size=self.parameters.input_size)
        model.to(self.device)

        # Optimizer
        optimizer = self.init_optimizer(model)

        # Train and evaluate
        trained_model = self.train_model(model, data_loader, data_loader_test, optimizer, on_epoch_end)

        # Save model
        if not os.path.isdir(self.parameters.output_folder):
            os.mkdir(self.parameters.output_folder)

        if not self.parameters.output_folder.endswith('/'):
            self.parameters.output_folder += '/'

        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        model_folder = self.parameters.output_folder + str_datetime + "/"

        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        # .pth
        if self.parameters.export_pth:
            model_path = model_folder + self.parameters.model_name + ".pth"
            ikutils.save_pth(trained_model, model_path)

        # .onnx
        if self.parameters.export_onnx:
            model_path = model_folder + self.parameters.model_name + ".onnx"
            input_shape = [1, 3, self.parameters.input_size, self.parameters.input_size]
            ikutils.save_onnx(trained_model, input_shape, self.device, model_path)

    def stop(self):
        self.stop_train = True
