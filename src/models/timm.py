import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb


class TimmModel(pl.LightningModule):
    """Expects a timm model as an input."""

    def __init__(self, model, num_classes=5, learning_rate=1e-2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.num_classes = num_classes
        self.model = model

        self.loss = F.cross_entropy
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)

    def forward(self, x):
        # TODO: implement fastai model structure: backbone --> head so we can use self.num_classes to set the head output
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        images, target = batch["image"], batch["target"]
        logits = self.forward(images)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('train_acc', acc, on_epoch=True, logger=True)
        self.log('conf_matrix', wandb.plot.confusion_matrix(probs=None,
                                                            preds=preds.cpu(), y_true=target.cpu(),
                                                            class_names=["Plantation", "Grassland",
                                                                         "Smallholder Agriculture"]))

        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch["image"], batch["target"]
        logits = self.forward(images)
        loss = self.loss(logits, target)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        self.log('val_loss', loss, on_step=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, logger=True)
        self.log('conf_matrix', wandb.plot.confusion_matrix(probs=None,
                                                            preds=preds.cpu(), y_true=target.cpu(),
                                                            class_names=["Plantation", "Grassland",
                                                                         "Smallholder Agriculture"]))

        return loss

    def test_step(self, batch, batch_idx):
        images, target = batch["image"], batch["target"]
        logits = self.forward(images)
        loss = self.loss(logits, target)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.05)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
