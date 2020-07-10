from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
from .datasets import MelanomaDataset
from .transforms import ImageTransform
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class MelanomaSystem(pl.LightningModule):
    def __init__(self, net, cfg, img_paths, df):
        super(MelanomaSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.img_paths = img_paths
        self.df = df
        self.train_dataset = None
        self.val_dataset = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_loss = 1e+9

    def prepare_data(self):
        # train_valid_split
        # StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        self.df['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(cv.split(self.df, self.df['target'])):
            self.df.loc[val_idx, 'fold'] = i

        fold = 0
        train_idx = np.where((self.df['fold'] != fold))[0]
        valid_idx = np.where((self.df['fold'] == fold))[0]
        train = self.df.iloc[train_idx]
        val = self.df.iloc[valid_idx]
        transform = ImageTransform()

        self.train_dataset = MelanomaDataset(train, self.img_paths, transform, phase='train')
        self.val_dataset = MelanomaDataset(val, self.img_paths, transform, phase='val')

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
        self.logger.log_hyperparams(dict(self.cfg.exp))

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

        return {'loss': loss, 'logits': logits, 'labels': label}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        LOGITS = torch.stack([x['logits'] for x in outputs]).reshape((-1)).detach().cpu().numpy()
        LABELS = torch.stack([x['labels'] for x in outputs]).reshape((-1)).detach().cpu().numpy()

        print(LOGITS)
        print(LABELS)

        auc = roc_auc_score(y_true=LABELS, y_score=LOGITS)
        logs = {'train/loss': avg_loss.item(), 'train/auc': auc}
        self.logger.log_metrics(logs)

        return {'avg_loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        img, label = batch

        out = self.forward(img)
        logits = torch.sigmoid(out)
        val_loss = self.criterion(out, label.unsqueeze(1))
        val_logs = {'val/loss': val_loss.item()}
        self.logger.log_metrics(val_logs, batch_idx)

        return {'val_loss': val_loss, 'logits': logits, 'labels': label}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        LOGITS = torch.stack([x['logits'] for x in outputs]).reshape((-1)).detach().cpu().numpy()
        LABELS = torch.stack([x['labels'] for x in outputs]).reshape((-1)).detach().cpu().numpy()

        auc = roc_auc_score(y_true=LABELS, y_score=LOGITS)
        logs = {'val/loss': avg_loss.item(), 'val/epoch_auc': auc}
        self.logger.log_metrics(logs)

        # 重みの保存
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss
            filename = f'loss_{self.best_loss:.3f}_auc_{auc:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            self.logger.experiment.log_artifact(self.logger.run_id, filename)
            os.remove(filename)

        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



