from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor
from torch.nn import Module

from physlearn.names import DataType, LabelType

CONF_KEY = "configuration"
STATE_KEY = "state"


class Model(Module, ABC):
    """
    API for torch models. Allows all trainers to be compatible with all models by
    giving identity to the input and output tensors.
    """

    @abstractmethod
    def forward(self, x: Dict[DataType, Tensor]) -> Dict[LabelType, Tensor]:
        """
        The forward pass of the model. This method also implements the decision where
        does each kind of input enter the model.

        Args:
            x: The input, a dictionary keyed by data type that contains the data of a
            sample batch presented to the model.

        Returns: The output, a dictionary keyed by data type that contains the output of
         a forward pass on a sample batch presented to the model.
        """
        pass
