from typing import Dict, Sequence, Union

from torch import Tensor

from physlearn.apis.loss import LossResult
from physlearn.apis.models import Model
from physlearn.names import BatchMetricType, EpochMetricType, LabelType, LossType


class BatchResult:
    """
    A container for the results of a single batch training
    """

    def __init__(
        self,
        loss: LossResult,
        metrics: Dict[BatchMetricType, float],
        batch_size: int,
        y: Dict[LabelType, Tensor],
        y_pred: Dict[LabelType, Tensor],
    ):
        """
        Args:
            loss: Batch loss
            metrics: Batch metrics
            batch_size: Number of samples in the batch
            y: Batch labels
            y_pred: Batch model predictions
        """
        self._metrics = metrics
        self._loss = loss
        self._batch_size = batch_size
        self.y = y
        self.y_pred = y_pred

    def __getitem__(self, item: Union[BatchMetricType, LossType]):
        if isinstance(item, BatchMetricType):
            return self.metrics[item]
        if isinstance(item, LossType):
            return self.loss if self.loss.type == item else None

    @property
    def metrics(self):
        return self._metrics

    @property
    def loss(self):
        return self._loss

    @property
    def batch_size(self):
        return self._batch_size


class EpochResult:
    """
    A container for the results of a single epoch training
    """

    def __init__(
        self,
        batch_losses: Sequence[LossResult],
        batch_metrics: Dict[BatchMetricType, Sequence[float]],
        epoch_metrics: Dict[EpochMetricType, float],
        epoch_size: int,
    ):
        """
        Args:
            batch_losses: A list of the losses of all batches in the epoch
            batch_metrics: Batch metrics for all batches in the epochs
            epoch_metrics: Aggregated epoch metrics
            epoch_size: Total number of samples in the epoch
        """
        self._batch_losses = batch_losses
        self._batch_metrics = batch_metrics
        self._epoch_metrics = epoch_metrics
        self._epoch_size = epoch_size

    def __getitem__(self, item: Union[BatchMetricType, EpochMetricType, LossType]):
        if isinstance(item, BatchMetricType):
            return self.batch_metrics[item]
        if isinstance(item, EpochMetricType):
            return self.epoch_metrics[item]

    @property
    def batch_losses(self):
        return self._batch_losses

    @property
    def batch_metrics(self):
        return self._batch_metrics

    @property
    def epoch_metrics(self):
        return self._epoch_metrics

    @property
    def epoch_size(self):
        return self._epoch_size


class FitResult:
    """
    A container for the results of an entire trainer model fit.
    """

    def __init__(
        self,
        batch_train_losses: Dict[int, Sequence[LossResult]],
        batch_validation_losses: Dict[int, Sequence[LossResult]],
        batch_train_metrics: Dict[BatchMetricType, Dict[int, Sequence[float]]],
        batch_validation_metrics: Dict[BatchMetricType, Dict[int, Sequence[float]]],
        epoch_train_metrics: Dict[EpochMetricType, Dict[int, float]],
        epoch_validation_metrics: Dict[EpochMetricType, Dict[int, float]],
        num_epochs: int,
        model: Model,
    ):
        """
        Args:
            batch_train_losses: Train losses per batch in a dictionary keyed by epoch
            number.
            batch_validation_losses: Validation losses per batch in a dictionary keyed
            by epoch number
            batch_train_metrics: Train metrics per batch, keyed by metric type stored in
            a dictionary keyed by epoch number.
            batch_validation_metrics: Validation metrics per batch, keyed by metric type
            stored in a dictionary keyed by epoch number.
            epoch_train_metrics: Train epoch aggregated metrics  in a dictionary keyed
            by epoch number.
            epoch_validation_metrics: Validation epoch aggregated metrics  in a
            dictionary keyed by epoch number.
            num_epochs: Total number of epochs the model was trained for.
            model: The trained model
        """
        self._batch_train_losses = batch_train_losses
        self._batch_validation_losses = batch_validation_losses
        self._batch_train_metrics = batch_train_metrics
        self._batch_validation_metrics = batch_validation_metrics
        self._epoch_train_metrics = epoch_train_metrics
        self._epoch_validation_metrics = epoch_validation_metrics
        self._num_epochs = num_epochs
        self._model = model

    @property
    def batch_train_losses(self):
        return self._batch_train_losses

    @property
    def batch_validation_losses(self):
        return self._batch_validation_losses

    @property
    def batch_train_metrics(self):
        return self._batch_train_metrics

    @property
    def batch_validation_metrics(self):
        return self._batch_validation_metrics

    @property
    def epoch_train_metrics(self):
        return self._epoch_train_metrics

    @property
    def epoch_validation_metrics(self) -> Dict[EpochMetricType, Dict[int, float]]:
        return self._epoch_validation_metrics

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def model(self):
        return self._model
