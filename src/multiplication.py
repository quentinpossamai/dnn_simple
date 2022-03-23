import argparse
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import Callable, Dict
import wandb

import src.utils.path as path


class MLP(pl.LightningModule):
    def __init__(self, num_in: int, num_out: int, num_layer: int, num_unit: int, activation: Callable,
                 lr: float):
        super(MLP, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.num_layer = num_layer
        self.num_unit = num_unit
        self.activation = activation
        self.lr = lr

        if self.num_layer == 1:
            self.linear_out = nn.Linear(self.num_in, self.num_out)
        else:
            assert self.num_layer >= 2, 'num_layer must be >= 1'
            self.linear0 = nn.Linear(self.num_in, self.num_unit)
            for i in range(1, self.num_layer - 1):
                exec(f'''self.linear{i} = nn.Linear(self.num_unit, self.num_unit)''')
            self.linear_out = nn.Linear(self.num_unit, self.num_out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layer == 1:
            return self.linear_out(x)
        else:
            out = self.activation(self.linear0(x))
            for i in range(1, self.num_layer - 1):
                exec(f'''out = self.activation(self.linear{i}(out))''')
            return self.linear_out(out)

    def step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :param batch: ['input', 'output']
        """
        return ((batch['output'] - self.forward(batch['input'])) ** 2).mean()

    def training_step(self, batch, batch_idx):
        training_loss = self.step(batch)
        self.log('training_loss', training_loss)
        return training_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.step(batch)
        self.log('val_loss', val_loss)
        return val_loss


class MultiplicationDataset(Dataset):
    def __init__(self, train_val_test: str):
        assert train_val_test in ['train', 'val', 'test']
        self.train_val_test = train_val_test

        self.path = f'{path.DATA}multiplication/'
        self.data = np.load(f'{self.path}{train_val_test}.npy')  # np array (n, 3), x1, x2, and x1x2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {'input': torch.from_numpy(self.data[item][0:2]).float(),
                'output': torch.from_numpy(self.data[item][2:3]).float()}

    def gen_data(self):
        x1 = np.random.randn(int(60e6))
        x2 = np.random.randn(int(60e6))
        np.save(f'{self.path}train.npy', np.stack([x1, x2, x1 * x2], axis=-1))

        x1 = np.random.randn(int(20e6))
        x2 = np.random.randn(int(20e6))
        np.save(f'{self.path}val.npy', np.stack([x1, x2, x1 * x2], axis=-1))

        x1 = np.random.randn(int(20e6))
        x2 = np.random.randn(int(20e6))
        np.save(f'{self.path}test.npy', np.stack([x1, x2, x1 * x2], axis=-1))


def training():
    seed_everything(0)
    analysis_path = f'{path.ANALYSIS}multiplication/'
    path.mkdir([analysis_path])

    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Identify model on data.')
    parser.add_argument('--activation', default='relu', type=str, help='Activation function')
    parser.add_argument('--batch_size', default=20, type=int, help='Size of a batch')
    parser.add_argument('--layer', default=2, type=int, help='Number of layers')
    parser.add_argument('--lr', default=-6, type=int, help="Learning rate")
    parser.add_argument('--unit', default=2, type=int, help='Size hidden layers')
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args accelerator, max_epochs
    args = parser.parse_args()

    # hyper parameters
    hp = {
        'activation': 'relu',
        'batch_size': args.batch_size,
        'layer': args.layer,
        'unit': args.unit,
        'lr': 10 ** args.lr,
    }
    activation = {'relu': torch.relu, 'tanh': torch.tanh}

    identifier = f'''Multiplication layer: {hp['layer']} unit: {hp['unit']}'''
    wandb.init(name=identifier,
               # id=identifier,
               project='dnn_simple',
               dir=analysis_path,
               config=hp)
    logger = WandbLogger(experiment=wandb.run)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = MLP(2, 1, hp['layer'], hp['unit'], activation[hp['activation']], hp['lr'])
    logger.watch(model)

    train_dataset = MultiplicationDataset('train')
    train_data = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)

    val_dataset = MultiplicationDataset('train')
    val_data = DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)

    checkpoint_callback = ModelCheckpoint(dirpath=analysis_path,
                                          filename='{epoch}-{step}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          mode='min')
    trainer = pl.Trainer(gpus=1 if device.type != 'cpu' else None,
                         logger=logger,
                         deterministic=True,
                         max_epochs=args.max_epochs,
                         # fast_dev_run=False,
                         # track_grad_norm=2,
                         # accelerator=None,
                         log_every_n_steps=50,  # default  50
                         accelerator='ddp' if device != 'cpu' else None,
                         plugins=DDPPlugin(find_unused_parameters=False),
                         log_gpu_memory='False',
                         resume_from_checkpoint=args.resume_from_checkpoint,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_data, val_data)
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    # MultiplicationDataset('train').gen_data()
    training()
