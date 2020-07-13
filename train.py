import os
import glob
import pandas as pd
import hydra
from omegaconf import DictConfig

from src.lightning import MelanomaSystem
from src.models import TestNet
from src.utils import seed_everything
from pytorch_lightning import Trainer
from comet_ml import Experiment

from pytorch_lightning.callbacks import ModelCheckpoint
from src.transforms import ImageTransform


@hydra.main('config.yml')
def main(cfg: DictConfig):
    data_dir = './input'
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    seed_everything(cfg.train.seed)
    # Comet.ml
    experiment = Experiment(api_key='LSTIie51umcysQtnef1Zzil6V', project_name='siim')

    # Load Data  ################################################################
    # train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train = pd.read_csv(os.path.join(data_dir, '_Alex_Dataset', 'folds.csv'),
                        usecols=['image_id', 'fold', 'target'])
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    img_paths = {
        'train': glob.glob(os.path.join(data_dir, '_Alex_Dataset',
                                        '512x512-dataset-melanoma', '512x512-dataset-melanoma', '*.jpg')),
        'test': glob.glob(os.path.join(data_dir, '_Alex_Dataset',
                                       '512x512-test', '512x512-test', '*.jpg')),
    }

    # Model  ####################################################################
    net = TestNet()
    transform = ImageTransform(img_size=cfg.data.img_size)

    # Lightning Module  #########################################################
    model = MelanomaSystem(net, cfg, img_paths, train, test, transform, experiment)

    checkpoint_callback = ModelCheckpoint(
        filepath='./checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=cfg.exp.exp_name + '_'
    )

    trainer = Trainer(
        max_epochs=cfg.train.epoch,
        checkpoint_callback=checkpoint_callback,
        gpus=[0]
        )

    # Train & Test  ############################################################
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    main()
