import os
import glob
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from src.lightning import MelanomaSystem
from src.models import TestNet, ENet
from src.utils import seed_everything
from pytorch_lightning import Trainer
from comet_ml import Experiment

from pytorch_lightning.callbacks import ModelCheckpoint
from src.transforms import ImageTransform, ImageTransform_2

import warnings
warnings.filterwarnings('ignore')

test_num = 20


@hydra.main('config.yml')
def main(cfg: DictConfig):
    data_dir = './input'
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    seed_everything(cfg.train.seed)
    # Comet.ml
    experiment = Experiment(api_key='LSTIie51umcysQtnef1Zzil6V', project_name='siim')

    # Load Data  ################################################################
    # Alex Dataset
    # train = pd.read_csv(os.path.join(data_dir, '_Alex_Dataset', 'folds.csv'),
    #                     usecols=['image_id', 'fold', 'target'])
    # test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    #
    # img_paths = {
    #     'train': glob.glob(os.path.join(data_dir, '_Alex_Dataset',
    #                                     '512x512-dataset-melanoma', '512x512-dataset-melanoma', '*.jpg')),
    #     'test': glob.glob(os.path.join(data_dir, '_Alex_Dataset',
    #                                    '512x512-test', '512x512-test', '*.jpg')),
    # }

    # Chris Dataset
    chris_image_size = 192
    _data_dir = f'./input/_Chris_Dataset_{chris_image_size}'
    train = pd.read_csv(os.path.join(_data_dir, 'train.csv'), usecols=['image_name', 'target'])
    # StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.train.seed)
    train['fold'] = -1
    for i, (trn_idx, val_idx) in enumerate(cv.split(train, train['target'])):
        train.loc[val_idx, 'fold'] = i

    test = pd.read_csv(os.path.join(_data_dir, 'test.csv'))

    img_paths = {
        'train': glob.glob(os.path.join(_data_dir, 'train', '*.jpg')),
        'test': glob.glob(os.path.join(_data_dir, 'test', '*.jpg'))
    }

    # Model  ####################################################################
    net = ENet(model_name=cfg.train.model_name)
    transform = ImageTransform_2(img_size=cfg.data.img_size, input_res=chris_image_size)

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
        # resume_from_checkpoint='./checkpoint/enet_b2_4_epoch=9.ckpt',
        max_epochs=cfg.train.epoch,
        checkpoint_callback=checkpoint_callback,
        gpus=[0]
        )

    # Train & Test  ############################################################
    # Train
    trainer.fit(model)

    # Test
    for i in range(test_num):
        trainer.test(model)

    # Submit
    sub_list = glob.glob('submission*.csv')
    res = pd.DataFrame()
    for i, path in enumerate(sub_list):
        sub = pd.read_csv(path)

        if i == 0:
            res['image_name'] = sub['image_name']
            res['target'] = sub['target']
        else:
            res['target'] += sub['target']

        os.remove(path)

    # min-max norm
    res['target'] -= res['target'].min()
    res['target'] /= res['target'].max()
    filename = 'submission.csv'
    res.to_csv(filename, index=False)
    experiment.log_asset(file_data=filename, file_name=filename)
    os.remove(filename)

    checkpoint_path = glob.glob(f'./checkpoint/{cfg.exp.exp_name}_*.ckpt')[0]
    experiment.log_asset(file_data=checkpoint_path)


if __name__ == '__main__':
    main()
