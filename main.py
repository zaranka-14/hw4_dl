import pandas as pd

from task_1 import RMSNorm
from task_2 import MyAutoGrad
from task_3 import *
from torch import nn
from torch.autograd import gradcheck

import torch


class Config:
    epochs: int = 10,
    lr: float = 0.01,
    base_hidden_size: int = 32,
    batch_size: int = 32,
    seed: int = 42,
    weight_decay: float = 0.9


if __name__ == '__main__':

    torch_norm = nn.RMSNorm([2, 3])
    my_norm = RMSNorm((2, 3))

    input_values = torch.randn(2, 2, 3)

    torch_res = torch_norm(input_values)
    my_res = my_norm(input_values)

    with open("results/task_1.txt", 'w') as f:
        f.write(f'Sum diff: {torch.sum(abs(torch_res - my_res))}\n')
        f.write(f'Max diff: {torch.max(abs(torch_res - my_res))}')

    x = torch.randn(5, requires_grad=True, dtype=torch.double)
    y = torch.randn(5, requires_grad=True, dtype=torch.double)

    my_res_t2 = MyAutoGrad.apply(x, y)
    must_be = torch.exp(x) + torch.cos(y)

    with open("results/task_2.txt", 'w') as f:
        f.write(f'Is grad ok: {gradcheck(MyAutoGrad.apply, (x, y))}\n')
        f.write(f'Forward diff: {torch.max(abs(my_res_t2 - must_be))}')

    train_df = pd.read_csv('data/loan_train.csv')
    test_df = pd.read_csv('data/loan_test.csv')

    train_ds, test_ds = load_loan(train_df)
    conf = Config()
    with open('results/task_3.txt', 'a') as f:
        train(conf, SimpleModel, train_ds, test_ds, f)
