import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size: test_size + train_size],
        'valid': indices[test_size + train_size:]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        val_acc = accuracy_score(y_val, classifier.predict(x_val))
        test_acc = accuracy_score(y_test, classifier.predict(x_test))
        return {
            'val_acc': val_acc,
            'test_acc': test_acc,
        }
