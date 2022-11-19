import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


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
        self.f_score = torchmetrics.F1Score(num_classes=num_classes, average='macro')

    def forward(self, x, metadata):
        # TODO: implement fastai model structure: backbone --> head so we can use self.num_classes to set the head output
        # output = self.model(x)
        x = self.model.features(x)
        x = self.model.global_pool(x)
        x = torch.cat((x, metadata), dim=1)
        x = self.model.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, target, metadata = batch["image"], batch["target"], batch["metadata"]
        logits = self.forward(images, metadata)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        f_score = self.f_score(preds, target)
        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('train_acc', acc, on_epoch=True, logger=True)
        self.log('train_f_score', f_score, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target, metadata = batch["image"], batch["target"], batch["metadata"]
        logits = self.forward(images, metadata)
        loss = self.loss(logits, target)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        f_score = self.f_score(preds, target)
        self.log('val_loss', loss, on_step=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, logger=True)
        self.log('val_f_score', f_score, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, target, metadata = batch["image"], batch["target"], batch["metadata"]
        logits = self.forward(images, metadata)
        loss = self.loss(logits, target)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
