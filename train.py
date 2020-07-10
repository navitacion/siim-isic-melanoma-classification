import os
from pathlib import Path
import glob
import pandas as pd
import hydra
from omegaconf import DictConfig

from src.lightning import MelanomaSystem
from src.models import TestNet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.mlflow import MLFlowLogger


@hydra.main('config.yml')
def main(cfg: DictConfig):
    data_dir = './input'
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # DownSampling
    temp = train[train['target'] == 1].reset_index(drop=True)
    train = train[train['target'] == 0].sample(n=1200).reset_index(drop=True)
    train = pd.concat([train, temp], axis=0, ignore_index=True)
    train = train.sample(frac=1.0).reset_index(drop=True)

    img_paths = glob.glob(os.path.join(data_dir, 'train', '*.dcm'))
    net = TestNet()
    logger = MLFlowLogger(experiment_name='test')
    model = MelanomaSystem(net, cfg, img_paths, train)
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        # gpus=[0]
        )

    trainer.fit(model)


if __name__ == '__main__':
    main()
