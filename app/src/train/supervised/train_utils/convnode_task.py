import pytorch_lightning as pl
from torchmetrics.classification import *
from model_utils import conv_models
import torch.nn.functional as F
import torch


class UCMERCDNeuralODE(pl.LightningModule):
    def __init__(self, num_filters = 64, augment_dim = True, time_dependent = True, non_linearity = 'relu', num_classes=10):
        super(UCMERCDNeuralODE, self).__init__()
        
        img_size = (3, 224, 224)
        output_dim = num_classes
        device = None

        # Define validation metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_mAP = MulticlassAveragePrecision(num_classes=num_classes, average="macro")
        self.val_math_corr_coeff = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_cohen_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # Define test metrics
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.model = conv_models.ConvODENet(device, img_size, num_filters,
                                output_dim=output_dim,
                                augment_dim=augment_dim,
                                time_dependent=time_dependent,
                                non_linearity=non_linearity,
                                adjoint=True)

    def forward(self, x):
        return self.model(x,return_features = False)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['image'],batch['label']
        # print(inputs.shape)
        # print(labels.shape)

        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        # preds = torch.argmax(outputs, dim=1)
        preds = outputs
        # acc = self.train_accuracy(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['image'],batch['label']
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        # print(labels.shape,labels)
        # preds = torch.argmax(outputs,dim=1)
        preds=outputs
        # print(preds.shape,preds)
        acc = self.val_accuracy(preds, labels)
        precision = self.val_precision(preds, labels)
        recall = self.val_recall(preds, labels)
        f1 = self.val_f1(preds, labels)
        mAP = self.val_mAP(preds, labels)
        math_corr_coeff = self.val_math_corr_coeff(preds, labels)
        cohen_kappa = self.val_cohen_kappa(preds, labels)
        cm = self.confusion_matrix(preds, labels)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True)
        self.log('val_mAP', mAP, on_step=False, on_epoch=True)
        self.log('val_math_corr_coeff', math_corr_coeff, on_step=False, on_epoch=True)
        self.log('val_cohen_kappa', cohen_kappa, on_step=False, on_epoch=True)

        return {'val_loss': loss, 'val_acc': acc, 'val_precision': precision, 
                'val_recall': recall, 'val_f1': f1, 'val_mAP': mAP, 
                'val_math_corr_coeff': math_corr_coeff, 'val_cohen_kappa': cohen_kappa, 'confusion_matrix': cm}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch['image'],batch['label']
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        # print(labels.shape,labels)
        # preds = torch.argmax(outputs,dim=1)
        preds=outputs
        # print(preds.shape,preds)
        acc = self.test_accuracy(preds, labels)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return {'test_loss': loss, 'test_acc': acc
               }

    # def on_validation_epoch_end(self):
    #     # Log confusion matrix at the end of each epoch
    #     confusion_matrix = self.confusion_matrix.compute()
    #     self.log("confusion_matrix", confusion_matrix)
    #     print(f"Confusion Matrix:\n{confusion_matrix}")
    #     self.confusion_matrix.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)