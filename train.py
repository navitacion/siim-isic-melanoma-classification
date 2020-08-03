import os
import glob
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

from src.lightning import MelanomaSystem, MelanomaSystem_2
from src.models import TestNet, ENet, ENet_2
from src.utils import seed_everything
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from comet_ml import Experiment

from pytorch_lightning.callbacks import ModelCheckpoint
from src.transforms import ImageTransform, ImageTransform_2, ImageTransform_3
from src.datasets import MelanomaDataset
from src.utils import summarize_submit, preprocessing_meta

import warnings
warnings.filterwarnings('ignore')

test_num = 20
is_multimodal = True


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    seed_everything(cfg.train.seed)
    # Comet.ml
    experiment = Experiment(api_key='LSTIie51umcysQtnef1Zzil6V', project_name='siim', log_code=True)

    # Load Data  ################################################################
    # Chris Dataset
    chris_image_size = cfg.data.load_size
    data_dir = f'./input/_Chris_Dataset_{chris_image_size}'
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    img_paths = {
        'train': glob.glob(os.path.join(data_dir, 'train', '*.jpg')),
        'test': glob.glob(os.path.join(data_dir, 'test', '*.jpg'))
    }

    # Cross Validation  #########################################################
    # StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.train.seed)
    train['fold'] = -1
    for i, (trn_idx, val_idx) in enumerate(cv.split(train, train['target'])):
        train.loc[val_idx, 'fold'] = i

    # GroupKFold
    cv = GroupKFold(n_splits=5)
    train['fold'] = -1
    for i, (trn_idx, val_idx) in enumerate(cv.split(train, train['target'], groups=train['patient_id'].tolist())):
        train.loc[val_idx, 'fold'] = i

    # Preprocessing  ############################################################
    # Drop Image
    # label=0なのにpred>0.5であった画像を除外する
    drop_image_name = ['ISIC_4579531', 'ISIC_7918608', 'ISIC_0948240', 'ISIC_4904364', 'ISIC_8780369', 'ISIC_8770180',
                       'ISIC_7148656', 'ISIC_7408392', 'ISIC_9959813', 'ISIC_1894141', 'ISIC_6633174', 'ISIC_3001941',
                       'ISIC_4259290', 'ISIC_6833905', 'ISIC_7452152', 'ISIC_2744859', 'ISIC_5464206', 'ISIC_6596403',
                       'ISIC_0711790', 'ISIC_5644568', 'ISIC_5843094', 'ISIC_8904326', 'ISIC_4963405', 'ISIC_9839042',
                       'ISIC_1355907', 'ISIC_0694037', 'ISIC_9513918', 'ISIC_0787851', 'ISIC_2932886', 'ISIC_2336763',
                       'ISIC_4064330', 'ISIC_7358293', 'ISIC_5789052', 'ISIC_7828320', 'ISIC_8277969', 'ISIC_1080647',
                       'ISIC_3238159', 'ISIC_8480913', 'ISIC_3790692', 'ISIC_0612624', 'ISIC_1242543', 'ISIC_4036915',
                       'ISIC_8174647', 'ISIC_2956783', 'ISIC_3302289', 'ISIC_6761105', 'ISIC_2152755', 'ISIC_9169000',
                       'ISIC_6852275', 'ISIC_4432898', 'ISIC_5459207', 'ISIC_7418664', 'ISIC_5136612', 'ISIC_9174738',
                       'ISIC_3160301', 'ISIC_7140636', 'ISIC_7718384', 'ISIC_9336675', 'ISIC_4282719', 'ISIC_4330005',
                       'ISIC_9828463', 'ISIC_6511141', 'ISIC_5335139', 'ISIC_5104921', 'ISIC_0695575', 'ISIC_0610141',
                       'ISIC_5946998', 'ISIC_0464315', 'ISIC_6556513', 'ISIC_3688407', 'ISIC_7730443', 'ISIC_4358550',
                       'ISIC_6461484', 'ISIC_9690422', 'ISIC_5374076', 'ISIC_1793200', 'ISIC_1389620', 'ISIC_8098274',
                       'ISIC_6425888', 'ISIC_6321076', 'ISIC_4298309', 'ISIC_2981912', 'ISIC_3650938', 'ISIC_4288522',
                       'ISIC_9459785', 'ISIC_1938535', 'ISIC_5576241', 'ISIC_6567889', 'ISIC_2768800', 'ISIC_6023795',
                       'ISIC_9281339', 'ISIC_6712494', 'ISIC_1811256', 'ISIC_5157055', 'ISIC_3943097', 'ISIC_7194471',
                       'ISIC_0361529', 'ISIC_9797578', 'ISIC_3575926', 'ISIC_6166824', 'ISIC_8828670', 'ISIC_6953126',
                       'ISIC_4430815', 'ISIC_8146054', 'ISIC_9305209', 'ISIC_4263017', 'ISIC_9314144', 'ISIC_1330763',
                       'ISIC_4792936', 'ISIC_1823608', 'ISIC_4910683', 'ISIC_9360142', 'ISIC_2863809', 'ISIC_4748668',
                       'ISIC_5681315', 'ISIC_3202829', 'ISIC_3450978', 'ISIC_9704624', 'ISIC_4350914', 'ISIC_3587744',
                       'ISIC_8190321', 'ISIC_1766413', 'ISIC_2872769', 'ISIC_3186625', 'ISIC_0170059', 'ISIC_4858099',
                       'ISIC_0314462', 'ISIC_2811886', 'ISIC_2140099', 'ISIC_9514450', 'ISIC_1195354', 'ISIC_8325872',
                       'ISIC_0227038', 'ISIC_6342641', 'ISIC_4162828', 'ISIC_7597293', 'ISIC_5278307', 'ISIC_3774190',
                       'ISIC_2957196', 'ISIC_4443545', 'ISIC_3455136', 'ISIC_0610499', 'ISIC_8483008', 'ISIC_0243683',
                       'ISIC_9028131', 'ISIC_8507102', 'ISIC_7128535', 'ISIC_4085552', 'ISIC_2940763', 'ISIC_1219894',
                       'ISIC_1043313', 'ISIC_6587979', 'ISIC_7050773', 'ISIC_3230164', 'ISIC_5159557', 'ISIC_7854457',
                       'ISIC_2582493', 'ISIC_5161114', 'ISIC_5238910', 'ISIC_6515221', 'ISIC_7771339', 'ISIC_9274260',
                       'ISIC_8054626', 'ISIC_1178847', 'ISIC_0236778', 'ISIC_6704518', 'ISIC_4214813', 'ISIC_0322818',
                       'ISIC_0230209', 'ISIC_7682938', 'ISIC_1852500', 'ISIC_3699454', 'ISIC_4693693', 'ISIC_9574591',
                       'ISIC_3465766', 'ISIC_1826803', 'ISIC_6234881', 'ISIC_2417958', 'ISIC_8142203', 'ISIC_5019268',
                       'ISIC_3251719', 'ISIC_4654808', 'ISIC_1027856', 'ISIC_3262153', 'ISIC_4681838', 'ISIC_6594555',
                       'ISIC_8623291', 'ISIC_3167092', 'ISIC_8791163', 'ISIC_1538510', 'ISIC_3962218', 'ISIC_2160145',
                       'ISIC_7690654', 'ISIC_9464203', 'ISIC_4673844', 'ISIC_9481260', 'ISIC_5407240', 'ISIC_5179742',
                       'ISIC_8851901', 'ISIC_7433711', 'ISIC_5777548', 'ISIC_2164933', 'ISIC_7194695', 'ISIC_7115605',
                       'ISIC_7560157', 'ISIC_1323909', 'ISIC_0307958', 'ISIC_8015259', 'ISIC_3089729', 'ISIC_3048886',
                       'ISIC_0861066', 'ISIC_6110309', 'ISIC_9103289', 'ISIC_2853454', 'ISIC_1436572', 'ISIC_9650546',
                       'ISIC_8208962', 'ISIC_5218561', 'ISIC_3285862', 'ISIC_5361506', 'ISIC_8196660', 'ISIC_0356238',
                       'ISIC_1156392', 'ISIC_2761440', 'ISIC_0645462', 'ISIC_4908514', 'ISIC_1374795', 'ISIC_3481768',
                       'ISIC_2102371', 'ISIC_4548990', 'ISIC_7200676', 'ISIC_8827725', 'ISIC_0667149', 'ISIC_7028320',
                       'ISIC_5485142', 'ISIC_9698871', 'ISIC_7764481', 'ISIC_8831706', 'ISIC_4478276', 'ISIC_0401250',
                       'ISIC_6987824', 'ISIC_7789537', 'ISIC_1114860', 'ISIC_7586566', 'ISIC_0343061', 'ISIC_1442157',
                       'ISIC_9161937', 'ISIC_5904214', 'ISIC_8335489', 'ISIC_9994768', 'ISIC_4384331', 'ISIC_0639415',
                       'ISIC_0982984', 'ISIC_2195070', 'ISIC_9022865', 'ISIC_0159060', 'ISIC_4933735', 'ISIC_3571989',
                       'ISIC_8593130', 'ISIC_1585919', 'ISIC_3907656', 'ISIC_9728805', 'ISIC_6029052', 'ISIC_3582787',
                       'ISIC_2205007', 'ISIC_1447559']
    train = train[~train['image_name'].isin(drop_image_name)].reset_index(drop=True)

    # Preprocessing metadata
    # OneHotEncoder
    train, test = preprocessing_meta(train, test)
    features_num = len([f for f in train.columns if f not in ['image_name', 'patient_id', 'target']])

    # Model  ####################################################################
    if is_multimodal:
        net = ENet_2(model_name=cfg.train.model_name, meta_features_num=features_num)
    else:
        net = ENet(model_name=cfg.train.model_name)
    transform = ImageTransform_3(img_size=cfg.data.img_size, input_res=chris_image_size)

    # Lightning Module  #########################################################
    if is_multimodal:
        model = MelanomaSystem_2(net, cfg, img_paths, train, test, transform, experiment)
    else:
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
        # resume_from_checkpoint='./checkpoint/enet_b5_1_512_epoch=4.ckpt',
        max_epochs=cfg.train.epoch,
        checkpoint_callback=checkpoint_callback,
        gpus=[0]
        )

    # Train & Test  ############################################################
    # Train
    trainer.fit(model)
    experiment.log_metric('best_auc', model.best_auc)
    checkpoint_path = glob.glob(f'./checkpoint/{cfg.exp.exp_name}_*.ckpt')[0]
    experiment.log_asset(file_data=checkpoint_path)

    # Test
    for i in range(test_num):
        trainer.test(model)

    # Submit
    sub_list = glob.glob(f'submission_{cfg.exp.exp_name}*.csv')
    _ = summarize_submit(sub_list, experiment, filename=f'submission_all_{cfg.exp.exp_name}.csv')

    # oof
    valid_dataset = MelanomaDataset(train, img_paths['train'], transform, phase='test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.train.batch_size, pin_memory=False,
                                  shuffle=False, drop_last=False)
    for i in range(10):
        trainer.test(model, test_dataloaders=valid_dataloader)

    # Submit
    sub_list = glob.glob('submission*.csv')
    res = summarize_submit(sub_list, experiment, filename=f'submission_oof_{cfg.exp.exp_name}.csv')

    # res = pd.concat([train, res], axis=1)


if __name__ == '__main__':
    main()
