from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from physlearn.apis.models import Model
from physlearn.apis.training.logging import Logger
from physlearn.apis.training.results import BatchResult, EpochResult, FitResult
from physlearn.names import CallbackType, DataType, LabelType, StopperType


class Callback:
    @abstractmethod
    def __call__(
        self,
        trainer: Trainer,
        train_result: EpochResult = None,
        validation_result: EpochResult = None,
    ):
        """
        Gets a trainer in a certain point during training and performs a desired
        action, e.g. plotting stats, saving checkpoints...

        Args:
            trainer: A trainer to operate upon.
            train_result: A train result to consider
            validation_result: A validation result to consider
        """
        pass

    @property
    @abstractmethod
    def type(self) -> CallbackType:
        """
        Returns: The type of callback
        """
        pass


class Stopper:
    @abstractmethod
    def __call__(
        self,
        trainer: Trainer,
        train_result: Optional[EpochResult] = None,
        validation_result: Optional[EpochResult] = None,
    ) -> bool:
        """
        Gets a trainer in a certain point during training and decides whether to stop
        the training.

        Args:
            trainer: A trainer to operate upon.
            train_result: A train result to consider
            validation_result: A validation result to consider

        Returns: Should the training be stopped?
        """
        pass

    @property
    @abstractmethod
    def type(self) -> StopperType:
        """
        Returns: The type of stopper
        """
        pass


class Trainer(ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    @abstractmethod
    def fit(
        self,
        dl_train: DataLoader,
        dl_val: DataLoader,
        stoppers: Sequence[Stopper],
        callbacks: Sequence[Callback] = (),
        print_every: int = 1,
        validation_every: int = 1,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.

        Args:
            dl_train: The training set data loader
            dl_val: The validation set data loader
            stoppers: A sequence of stoppers, each represent a situation on which the
            training stops (e.g. number of epochs is reached, early stopping condition)
            callbacks:
            print_every: Show a status bar in stdout i=once every this number of
            training epochs. Use only for online runs, for batch execution disable by
            setting to 0.
            validation_every: A number of training epochs after which a validation epoch
            is calculated.
        """
        pass

    @property
    @abstractmethod
    def logger(self) -> Logger:
        """
        Returns: The trainer logger, an object tracking the results, calculating
        metrics and creating the result objects. On certain implementations also
        does runtime logging.
        """
        pass

    @abstractmethod
    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).

        Args:
            dl_train: A training data loader
            **kw: Implementation specific training parameters

        Returns:

        """
        pass

    @abstractmethod
    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).

        Args:
            dl_test: A testing data loader
            **kw: Implementation specific training parameters

        Returns:

        """
        pass

    @abstractmethod
    def train_batch(
        self, batch: Tuple[Dict[DataType, Tensor], Dict[LabelType, Tensor]]
    ) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.

        Args:
            batch: A single batch of data from a data loader, a tuple with two
            dictionaries, one for the inputs and one for the labels.

        Returns: A batch result with all the batch metrics defined in the logger.
        """
        pass

    @abstractmethod
    def test_batch(
        self, batch: Tuple[Dict[DataType, Tensor], Dict[LabelType, Tensor]]
    ) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.

        Args:
            batch: A single batch of data  from a data loader, a tuple with two
            dictionaries, one for the inputs and one for the labels.

        Returns: A batch result with all the batch metrics defined in the logger.

        """
        pass

    @property
    @abstractmethod
    def model(self) -> Model:
        """
        Returns: The model being trained by the trainer
        """
        pass
