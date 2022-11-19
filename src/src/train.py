import pytorch_lightning as pl
import timm
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn


from src.callbacks.callbacks import ImagePredictionLogger
from src.data.datamodule import DataModule
from src.models.timm import TimmModel
from src.transforms.transforms import train_transforms, valid_transforms, test_transforms
from src.utils import lr_find


def main():
    # Init our data pipeline
    transforms = {
        "train": train_transforms,
        "valid": valid_transforms,
        "test": test_transforms
        }

    print("Initializing DataModule...")
    dm = DataModule(root_data_dir="./data/deforest/", batch_size=8, transforms=transforms)
    dm.setup()
    print("Done!")

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))

    # Init our model
    print("Creating Lightning Module with timm model...")
    num_classes = 3
    timm_model = timm.create_model('densenet201', pretrained=True, num_classes=num_classes)  #https://rwightman.github.io/pytorch-image-models/
    timm_model.classifier = nn.Linear(1923, 3) 
    model = TimmModel(timm_model, num_classes, learning_rate=1e-3)
    print("Done!")

    # Initialize wandb logger
    print("Initializing Wandb...")
    wandb_logger = WandbLogger(project='Deforestation-Classification', job_type='train')
    print("Done!")

    # Initialize callbacks
    callbacks = [
        #EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max"),
        LearningRateMonitor(),
        ImagePredictionLogger(val_samples),
        ModelCheckpoint(dirpath="./checkpoints", monitor="val_loss", filename="ship-{epoch:02d}-{val_loss:.2f}")
    ]

    # Initialize a trainer
    print("Initializing trainer...")
    trainer = pl.Trainer(max_epochs=30,
                         gpus=1,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         enable_progress_bar=True)
    print("Done!")

    # Find lr
    print("Finding optimal lr...")
    lr_find(trainer, model, dm)
    print("Done!")

    # Train the model ⚡🚅⚡
    print("Training...")
    trainer.fit(model, dm)
    print("Done!")

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
