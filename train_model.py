from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig
import hydra

from model import Model
from dataset import ClassificationDataModule


@hydra.main(config_name='train_cfg')
def main(cfg: DictConfig) -> None:

    dm = ClassificationDataModule(cfg)
    model = Model(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename='bs_{0}_lr_{1}'.format(cfg.batch_size, cfg.lr),
        verbose=cfg.verbose
    )

    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='max',
        patience=150,
        verbose=cfg.verbose,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, stop_callback],
        precision=16,
        accelerator='auto',
        max_epochs=1000
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(
        cfg.checkpoint_dir+'/conv2d_yoav.ckpt')


if __name__ == "__main__":
    main()
