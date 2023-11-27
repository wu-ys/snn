import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np

from pathlib import Path

from spikingjelly.datasets import n_mnist
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from tqdm import tqdm
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase

from power_surrogate import PowerSurrogate
from trigono_surrogate import TrigonoSurrogate


class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(2 * 34 * 34, 500, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function = TrigonoSurrogate(alpha=1.0)), # change surrogate func as you wish
            layer.Linear(500, 10, bias=False),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

root = Path("/mnt/d/wys/classfile/2023-2024-1/neuralAI/lab3")

T = 8
train_dataset = n_mnist.NMNIST(root=root,
                               train=True,
                               data_type='frame',
                               frames_number=T,
                               split_by='number',
            transform=None,
            target_transform=None)

test_dataset = n_mnist.NMNIST(root=root,
                               train=False,
                               data_type='frame',
                               frames_number=T,
                               split_by='number',
            transform=None,
            target_transform=None)

train_data_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

test_data_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)
net = SNN(tau=10.).cuda() #if you have a GPU
start_epoch = 0
max_test_acc = 0

optimizer = torch.optim.Adam(net.parameters(), lr=0.01) # do not forget to finetune these training parameters

total_epoch = 20

train_losses = []
train_accs = []
test_accs = []

for epoch in range(total_epoch):
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for frames, label in tqdm(train_data_loader):
        optimizer.zero_grad()
        frames = frames.cuda() #if you have a GPU
        label = label.cuda() #if you have a GPU
        label_onehot = F.one_hot(label, 10).float()

        output = 0.
        for t in range(T):
            output += net(frames[:, t])
        output = output / T
        loss = F.mse_loss(output, label_onehot)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (output.argmax(1) == label).float().sum().item()
        functional.reset_net(net)

    train_loss /= train_samples
    train_acc /= train_samples

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    print('Epoch %d/%d: train acc: %.3f' % (epoch, total_epoch, train_acc))

    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frames, label in tqdm(test_data_loader):
            frames = frames.cuda()# if you have a GPU
            label = label.cuda()# if you have a GPU
            label_onehot = F.one_hot(label, 10).float()
            output = 0.
            for t in range(T):
                output += net(frames[:, t])
            output = output / T
            loss = F.mse_loss(output, label_onehot)

            test_samples += label.numel()
            test_acc += (output.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    test_acc /= test_samples

    test_accs.append(test_acc)
    print('Epoch %d/%d: test acc: %.3f' % (epoch, total_epoch, test_acc))

data_dict = {
    'train_loss': train_losses,
    'train_acc': train_accs,
    'test_acc': test_accs
}

np.savez("data_")
