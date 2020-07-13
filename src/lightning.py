import pandas as pd
import itertools
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from .datasets import MelanomaDataset
from .transforms import ImageTransform
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.metrics.classification import AUROC
from torch_optimizer import RAdam


# Config  #######################
label_smoothing = 0.2


class MelanomaSystem(pl.LightningModule):
    def __init__(self, net, cfg, img_paths, train_df, test_df, transform, experiment):
        super(MelanomaSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.img_paths = img_paths
        self.train_df = train_df
        self.test_df = test_df
        self.transform = transform
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.experiment = experiment
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_loss = 1e+9
        self.epoch_num = 0

    def prepare_data(self):
        # Split Train, Validation
        fold = self.cfg.train.fold
        train = self.train_df[self.train_df['fold'] != fold].reset_index(drop=True)
        val = self.train_df[self.train_df['fold'] == fold].reset_index(drop=True)

        self.train_dataset = MelanomaDataset(train, self.img_paths['train'], self.transform, phase='train')
        self.val_dataset = MelanomaDataset(val, self.img_paths['train'], self.transform, phase='val')
        self.test_dataset = MelanomaDataset(self.test_df, self.img_paths['test'], self.transform, phase='test')

        # Save Hyper Parameters
        self.set_params()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          sampler=RandomSampler(self.train_dataset), drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          sampler=SequentialSampler(self.val_dataset), drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=False, drop_last=False)

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.cfg.train.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epoch_num,
                                      eta_min=self.cfg.train.lr * 0.01)
        return [optimizer], [scheduler]

    def set_params(self):
        # Log Parameters
        self.experiment.log_parameters(dict(self.cfg.exp))
        self.experiment.log_parameters(dict(self.cfg.data))
        self.experiment.log_parameters(dict(self.cfg.train))

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        inp, label, img_name = batch
        out = self.forward(inp)

        if label is not None:
            label_smo = label.float() * (1 - label_smoothing) + 0.5 * label_smoothing
            loss = self.criterion(out, label_smo.unsqueeze(1))
        else:
            loss = None

        return loss, label, torch.sigmoid(out), img_name

    def training_step(self, batch, batch_idx):
        loss, label, logits, _ = self.step(batch)

        logs = {'train/loss': loss.item()}
        self.experiment.log_metrics(logs, step=batch_idx)

        return {'loss': loss, 'logits': logits, 'labels': label}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, _ = self.step(batch)

        val_logs = {'val/loss': loss.item()}
        self.experiment.log_metrics(val_logs, step=batch_idx)

        return {'val_loss': loss, 'logits': logits.detach(), 'labels': label.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        LOGITS = torch.cat([x['logits'] for x in outputs])
        LABELS = torch.cat([x['labels'] for x in outputs])

        # Skip Sanity Check
        auc = AUROC()(pred=LOGITS, target=LABELS) if LABELS.float().mean() > 0 else 0.5
        logs = {'val/epoch_loss': avg_loss.item(), 'val/epoch_auc': auc}
        # Log loss, auc
        self.experiment.log_metrics(logs, step=self.epoch_num)
        self.epoch_num += 1

        # Save Weights
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss
            filename = f'{self.cfg.exp.exp_name}_epoch_{self.epoch_num}_loss_{self.best_loss:.3f}_auc_{auc:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            self.experiment.log_model(name=filename, file_or_folder='./'+filename)
            os.remove(filename)

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        loss, _, logits, img_name = self.step(batch)

        return {'preds': logits, 'image_names': img_name}

    def test_epoch_end(self, outputs):
        PREDS = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        # [tuple, tuple]
        IMG_NAMES = [x['image_names'] for x in outputs]
        # [list, list]
        IMG_NAMES = [list(x) for x in IMG_NAMES]
        IMG_NAMES = list(itertools.chain.from_iterable(IMG_NAMES))

        res = pd.DataFrame({
            'image_name': IMG_NAMES,
            'target': PREDS.tolist()
        })
        filename = 'submission.csv'
        res.to_csv(filename, index=False)
        self.experiment.log_asset(file_data=filename, file_name=filename)
        os.remove('submission.csv')

        return {'res': res}
