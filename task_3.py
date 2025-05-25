from torch.optim import Optimizer
from torch import nn, Tensor
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, AUROC
from tqdm import tqdm


class Lion(Optimizer):
    def __init__(self, params, weight=0.0, gradient=0.9,
                 momentum=0.9, lr=1e-4, betas=(0.9, 0.99)):
        super().__init__(params, {
            'weight': weight,
            'gradient': gradient,
            'momentum': momentum,
            'lr': lr,
            'betas': betas
        })

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight']

                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                momentum = state['momentum']

                update = (1 - beta1) * grad + beta1 * momentum
                update = torch.sign(update)

                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)

                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                p.add_(update, alpha=-lr)

        return loss


class Block(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.lin1 = nn.Linear(hidden_size, hidden_size * 4)
        self.r1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.r1(x)
        x = self.lin2(x)

        return x


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()

        self.emb_pho = nn.Embedding(4, embedding_dim=hidden_size)
        self.emb_intent = nn.Embedding(6, embedding_dim=hidden_size)
        self.emb_grade = nn.Embedding(7, embedding_dim=hidden_size)
        self.emb_cpdof = nn.Embedding(2, embedding_dim=hidden_size)

        self.num_lin = nn.Linear(7, hidden_size)

        self.block = Block(hidden_size)

        self.lin_to_out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, torch.Tensor], numeric_features: dict[str, torch.Tensor]) -> torch.Tensor:
        x_pho = self.emb_pho(cat_features['pho'])
        x_intent = self.emb_intent(cat_features['intent'])
        x_grade = self.emb_grade(cat_features['grade'])
        x_cpdof = self.emb_cpdof(cat_features['cpdof'])

        stacked_num = torch.stack([numeric_features['age'], numeric_features['income'],
                                   numeric_features['pel'], numeric_features['amnt'],
                                   numeric_features['rate'], numeric_features['pincome'],
                                   numeric_features['cbpch']], dim=-1)

        x_num = self.num_lin(stacked_num)
        x = x_num + x_pho + x_intent + x_grade + x_cpdof

        x = self.block(x)
        x = self.lin_to_out(x)
        x = x.squeeze(-1)

        return x


def train(conf, input_model, train_ds, eval_ds, file):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_epochs = conf.epochs[0]
    lr = conf.lr[0]
    base_hidden_size = conf.base_hidden_size[0]
    batch_size = conf.batch_size[0]
    seed = conf.seed[0]
    weight_decay = conf.weight_decay
    torch.random.manual_seed(seed)

    loss_bce = BCEWithLogitsLoss()

    collator = LoanCollator()
    model = input_model(hidden_size=base_hidden_size).to(dev)
    optimizer = Lion(model.parameters(), lr=lr, weight=weight_decay)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=2, collate_fn=collator, pin_memory=True)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, num_workers=2, collate_fn=collator, pin_memory=True)

    for epoch in tqdm(range(n_epochs)):
        train_loss = MeanMetric().to(dev)
        train_rocauc = AUROC(task='binary').to(dev)
        for i, batch in enumerate(train_dl):
            map_to_device(batch, dev)

            result = model(cat_features=batch['cat_features'], numeric_features=batch['numeric_features'])
            loss_value = loss_bce(result, batch['target'])
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss_value)
            train_rocauc.update(torch.sigmoid(result), batch['target'])

        train_loss = train_loss.compute().item()
        train_rocauc = train_rocauc.compute().item()
        file.write(f'train epoch: {epoch}, train_loss: {train_loss}, train_rocauc: {train_rocauc}\n')

        eval_loss = MeanMetric().to(dev)
        eval_rocauc = AUROC(task='binary').to(dev)

        model.eval()
        with torch.no_grad():
            for i_eval, batch_eval in enumerate(eval_dl):
                map_to_device(batch_eval, dev)

                result_eval = model(cat_features=batch_eval['cat_features'],
                                    numeric_features=batch_eval['numeric_features'])
                eval_loss_value = loss_bce(result_eval, batch_eval['target'])

                eval_loss.update(eval_loss_value)
                eval_rocauc.update(torch.sigmoid(result_eval), batch_eval['target'])
        model.train()

        eval_loss = eval_loss.compute().item()
        eval_rocauc = eval_rocauc.compute().item()
        file.write(f'eval epoch: {epoch}, train_loss: {eval_loss}, train_rocauc: {eval_rocauc}\n')


maps = {
    'pho_map': {
        'MORTGAGE': 0,
        'RENT': 1,
        'OWN': 2,
        'OTHER': 3
    },

    'intent_map': {
        'EDUCATION': 0,
        'HOMEIMPROVEMENT': 1,
        'MEDICAL': 2,
        'DEBTCONSOLIDATION': 3,
        'VENTURE': 4,
        'PERSONAL': 5
    },

    'grade_map': {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6
    },

    'cpdof_map': {
        'N': 0,
        'Y': 1
    }

}


def map_to_device(batch: dict, dev: torch.device):
    batch['target'] = batch['target'].to(dev)

    batch['cat_features']['pho'] = batch['cat_features']['pho'].to(dev)
    batch['cat_features']['intent'] = batch['cat_features']['intent'].to(dev)
    batch['cat_features']['grade'] = batch['cat_features']['grade'].to(dev)
    batch['cat_features']['cpdof'] = batch['cat_features']['cpdof'].to(dev)

    batch['numeric_features']['age'] = batch['numeric_features']['age'].to(dev)
    batch['numeric_features']['income'] = batch['numeric_features']['income'].to(dev)
    batch['numeric_features']['pel'] = batch['numeric_features']['pel'].to(dev)
    batch['numeric_features']['amnt'] = batch['numeric_features']['amnt'].to(dev)
    batch['numeric_features']['rate'] = batch['numeric_features']['rate'].to(dev)
    batch['numeric_features']['pincome'] = batch['numeric_features']['pincome'].to(dev)
    batch['numeric_features']['cbpch'] = batch['numeric_features']['cbpch'].to(dev)


class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int) -> dict[str, dict[str | Tensor] | Tensor]:
        item = self._data.iloc[item]
        return {
            'target': torch.scalar_tensor(item['loan_status'], dtype=torch.float32),
            'cat_features': {
                'pho': torch.scalar_tensor(maps['pho_map'][item['person_home_ownership']], dtype=torch.long),
                'intent': torch.scalar_tensor(maps['intent_map'][item['loan_intent']], dtype=torch.long),
                'grade': torch.scalar_tensor(maps['grade_map'][item['loan_grade']], dtype=torch.long),
                'cpdof': torch.scalar_tensor(maps['cpdof_map'][item['cb_person_default_on_file']], dtype=torch.long)
            },
            'numeric_features': {
                'age': torch.scalar_tensor(-1 if pd.isna(item['person_age'])
                                           else (item['person_age'] / 123), dtype=torch.float32),

                'income': torch.scalar_tensor(-1 if pd.isna(item['person_income'])
                                              else (item['person_income'] / 1900000), dtype=torch.float32),

                'pel': torch.scalar_tensor(-1 if pd.isna(item['person_emp_length'])
                                           else (item['person_emp_length'] / 123), dtype=torch.float32),

                'amnt': torch.scalar_tensor(-1 if pd.isna(item['loan_amnt'])
                                            else (item['loan_amnt'] / 35000), dtype=torch.float32),

                'rate': torch.scalar_tensor(-1 if pd.isna(item['loan_int_rate'])
                                            else (item['loan_int_rate'] / 24), dtype=torch.float32),

                'pincome': torch.scalar_tensor(-1 if pd.isna(item['loan_percent_income'])
                                               else (item['loan_percent_income']), dtype=torch.float32),

                'cbpch': torch.scalar_tensor(-1 if pd.isna(item['cb_person_cred_hist_length'])
                                             else (item['cb_person_cred_hist_length'] / 30), dtype=torch.float32),
            }
        }


class LoanCollator:
    def __call__(self, items: list[dict[str, dict[str | Tensor] | Tensor]]) -> dict[str, dict[str | Tensor] | Tensor]:
        return {
            'target': torch.stack([x['target'] for x in items]),
            'cat_features': {
                'pho': torch.stack([x['cat_features']['pho'] for x in items]),
                'intent': torch.stack([x['cat_features']['intent'] for x in items]),
                'grade': torch.stack([x['cat_features']['grade'] for x in items]),
                'cpdof': torch.stack([x['cat_features']['cpdof'] for x in items])
            },
            'numeric_features': {
                'age': torch.stack([x['numeric_features']['age'] for x in items]),
                'income': torch.stack([x['numeric_features']['income'] for x in items]),
                'pel': torch.stack([x['numeric_features']['pel'] for x in items]),
                'amnt': torch.stack([x['numeric_features']['amnt'] for x in items]),
                'rate': torch.stack([x['numeric_features']['rate'] for x in items]),
                'pincome': torch.stack([x['numeric_features']['pincome'] for x in items]),
                'cbpch': torch.stack([x['numeric_features']['cbpch'] for x in items])
            }
        }


def load_loan(df) -> tuple[LoanDataset, LoanDataset]:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['loan_status'])
    return LoanDataset(df_train), LoanDataset(df_test)
