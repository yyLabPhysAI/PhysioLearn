from abc import ABC, abstractmethod
from typing import Dict, Type

from torch import Tensor

from physlearn.names import LabelType, LossType


class LossResult(ABC):
    def __init__(self, value: float):
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    @abstractmethod
    def type(self) -> LossType:
        pass


class LossFunc(ABC):
    @abstractmethod
    def __call__(
        self, y: Dict[LabelType, Tensor], y_pred: Dict[LabelType, Tensor]
    ) -> Tensor:
        pass

    @property
    @abstractmethod
    def type(self) -> LossType:
        pass

    @classmethod
    @abstractmethod
    def result_class(cls) -> Type[LossResult]:
        pass
