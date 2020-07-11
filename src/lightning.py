from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from .datasets import MelanomaDataset
from .transforms import ImageTransform
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class MelanomaSystem(pl.LightningModule):
    def __init__(self, net, cfg, img_paths, train_df, experiment):
        super(MelanomaSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.img_paths = img_paths
        self.train_df = train_df
        self.train_dataset = None
        self.val_dataset = None
        self.experiment = experiment
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_loss = 1e+9
        self.epoch_num = 0

    def prepare_data(self):
        # StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        self.train_df['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(cv.split(self.train_df, self.train_df['target'])):
            self.train_df.loc[val_idx, 'fold'] = i

        fold = 0
        train_idx = np.where((self.train_df['fold'] != fold))[0]
        valid_idx = np.where((self.train_df['fold'] == fold))[0]
        train = self.train_df.iloc[train_idx]
        val = self.train_df.iloc[valid_idx]
        transform = ImageTransform()

        self.train_dataset = MelanomaDataset(train, self.img_paths['train'], transform, phase='train')
        self.val_dataset = MelanomaDataset(val, self.img_paths['train'], transform, phase='val')

        # ハイパーパラメータを設定
        self.set_params()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          sampler=RandomSampler(self.train_dataset), drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          sampler=SequentialSampler(self.val_dataset), drop_last=True)

    def set_params(self):
        # mlflow
        self.logger.log_hyperparams(dict(self.cfg.exp))
        self.logger.log_hyperparams(dict(self.cfg.train))
        # Comet_ml
        self.experiment.log_parameters(dict(self.cfg.exp))
        self.experiment.log_parameters(dict(self.cfg.train))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label = batch

        out = self.forward(img)
        logits = torch.sigmoid(out)
        loss = self.criterion(out, label.unsqueeze(1))
        # logはitemで値だけ入れないとだめっぽい
        logs = {'train/loss': loss.item()}
        self.logger.log_metrics(logs, batch_idx)
        self.experiment.log_metrics(logs, step=batch_idx)

        return {'loss': loss, 'logits': logits, 'labels': label}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        LOGITS = torch.stack([x['logits'] for x in outputs]).reshape((-1)).detach().cpu().numpy()
        LABELS = torch.stack([x['labels'] for x in outputs]).reshape((-1)).detach().cpu().numpy()

        auc = roc_auc_score(y_true=LABELS, y_score=LOGITS)
        logs = {'train/epoch_loss': avg_loss.item(), 'train/epoch_auc': auc}
        self.logger.log_metrics(logs)
        self.experiment.log_metrics(logs, step=self.epoch_num)

        return {'avg_loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        img, label = batch

        out = self.forward(img)
        logits = torch.sigmoid(out)
        val_loss = self.criterion(out, label.unsqueeze(1))
        val_logs = {'val/loss': val_loss.item()}
        self.logger.log_metrics(val_logs, batch_idx)
        self.experiment.log_metrics(val_logs, step=batch_idx)

        return {'val_loss': val_loss, 'logits': logits, 'labels': label}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        LOGITS = torch.stack([x['logits'] for x in outputs]).reshape((-1)).detach().cpu().numpy()
        LABELS = torch.stack([x['labels'] for x in outputs]).reshape((-1)).detach().cpu().numpy()

        auc = roc_auc_score(y_true=LABELS, y_score=LOGITS)
        logs = {'val/epoch_loss': avg_loss.item(), 'val/epoch_auc': auc}
        # MLFlow
        self.logger.log_metrics(logs)
        # Comet.ml
        self.experiment.log_metrics(logs, step=self.epoch_num)
        self.epoch_num += 1

        # 重みの保存
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss
            filename = f'{self.cfg.exp.exp_name}_epoch_{self.epoch_num}_loss_{self.best_loss:.3f}_auc_{auc:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            self.logger.experiment.log_artifact(self.logger.run_id, filename)
            self.experiment.log_model(name=filename, file_or_folder='./'+filename)
            os.remove(filename)

        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



