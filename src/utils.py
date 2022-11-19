import pytorch_lightning as pl


def lr_find(trainer: pl.Trainer, model, dm: pl.LightningDataModule):
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, dm)

    # Results can be found in
    print(lr_finder.results)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr
