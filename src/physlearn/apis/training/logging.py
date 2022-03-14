from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, cast

from torch import Tensor

from physlearn.apis.loss import LossResult
from physlearn.apis.models import Model
from physlearn.apis.training.metrics import BatchMetric, EpochMetric
from physlearn.apis.training.results import BatchResult, EpochResult, FitResult
from physlearn.names import BatchMetricType, EpochMetricType, LabelType


class Logger:
    """
    Logs the training and reports the results.

    This base class merely collects results and calculate metrics, full logging is
    implementation and environment dependent and should be implemented in inheriting
    classes.
    """

    def __init__(
        self,
        batch_metrics: Sequence[BatchMetric],
        epoch_metrics: Sequence[EpochMetric],
        start_epoch: int = 0,
        keep_output: bool = False,
    ):
        self._batch_metrics = batch_metrics
        self._epoch_metrics = epoch_metrics
        self._current_epoch = start_epoch
        self._keep_output = keep_output

        self._current_epoch_batch_results: Deque[BatchResult] = deque()

        self._batch_train_results: Dict[int, Deque] = {}
        self._batch_validation_results: Dict[int, Deque] = {}

        self._epoch_train_results: Dict[int, EpochResult] = {}
        self._epoch_validation_results: Dict[int, EpochResult] = {}

        self.mode: str = ""

        self.train_epochs_run = 0
        self.validation_epochs_run = 0

        self.fit_result: Optional[FitResult] = None

    def new_batch(
        self,
        loss: LossResult,
        y: Dict[LabelType, Tensor],
        y_pred: Dict[LabelType, Tensor],
        batch_size: int,
        logs: Optional[Dict] = None,
    ) -> BatchResult:
        """
        Reports a new batch result, invoked in the end of every batch.

        Args:
            loss: Batch loss, with the structure (loss type, loss)
            y: Batch labels
            y_pred: Batch model predictions
            batch_size: Number of samples in this batch
            logs: Additional logs collected during batch training

        Returns: A BatchResult containing the batch loss and all metrics
        """
        if self._keep_output:
            self._save_output(y, y_pred)

        batch_result = BatchResult(
            loss,
            {m.type: m(y, y_pred) for m in self._batch_metrics},
            batch_size,
            y,
            y_pred,
        )
        self._current_epoch_batch_results.append(batch_result)
        self._log_batch(logs, batch_result)

        return batch_result

    def start_epoch(self, train: bool, validation: bool, current_epoch: int):
        if train == validation:
            raise ValueError("Epoch can be either train or validation. Not both.")
        self._current_epoch_batch_results = deque()
        self.mode = "training" if train else "validation"
        self._current_epoch = current_epoch

    def finish_epoch(self, logs: Optional[Dict] = None) -> EpochResult:
        """
        Reports the end of an epoch. Accumulates all the batch results and calculated
        epoch accumulated results.

        Args:
            logs: Additional logs collected during epoch training

        Returns:  An EpochResult containing the losses and metrics for all batches in
        the epoch and aggregated epoch metrics.
        """

        batch_losses = []
        batch_metrics: Dict[BatchMetricType, List[float]] = {
            metric.type: [] for metric in self._batch_metrics
        }
        epoch_size: int = 0

        for batch in self._current_epoch_batch_results:
            batch_losses.append(batch.loss)
            for metric in self._batch_metrics:
                batch_metrics[metric.type].append(batch[metric.type])
            epoch_size += batch.batch_size

        epoch_metrics: Dict[EpochMetricType, float] = {}
        for m in self._epoch_metrics:
            epoch_metrics[m.type] = m(self._current_epoch_batch_results)

        epoch_result = EpochResult(
            batch_losses,
            cast(Dict[BatchMetricType, Sequence[float]], batch_metrics),
            epoch_metrics,
            epoch_size,
        )

        if self.mode == "training":
            self._batch_train_results[
                self._current_epoch
            ] = self._current_epoch_batch_results
            self._epoch_train_results[self._current_epoch] = epoch_result
            self.train_epochs_run += 1
        elif self.mode == "validation":
            self._batch_validation_results[
                self._current_epoch
            ] = self._current_epoch_batch_results
            self._epoch_validation_results[self._current_epoch] = epoch_result
            self.validation_epochs_run += 1
        else:
            raise RuntimeError(f"Unknown mode {self.mode}")

        self._log_epoch(logs, epoch_result)
        return epoch_result

    def finish_training(self, model: Model, logs: Optional[Dict] = None) -> FitResult:
        """

        Args:
            model: The trained model to add to the fit result
            logs: Additional logs collected during epoch training

        Returns:

        """

        batch_train_losses: Dict[int, Sequence[LossResult]] = {}
        batch_validation_losses: Dict[int, Sequence[LossResult]] = {}
        batch_train_metrics: Dict[BatchMetricType, Dict[int, Sequence[float]]] = {
            m.type: {} for m in self._batch_metrics
        }
        batch_validation_metrics: Dict[BatchMetricType, Dict[int, Sequence[float]]] = {
            m.type: {} for m in self._batch_metrics
        }
        epoch_train_metrics: Dict[EpochMetricType, Dict[int, float]] = {
            m.type: {} for m in self._epoch_metrics
        }
        epoch_validation_metrics: Dict[EpochMetricType, Dict[int, float]] = {
            m.type: {} for m in self._epoch_metrics
        }

        for epoch, res in self._epoch_train_results.items():
            batch_train_losses[epoch] = res.batch_losses
            for batch_metric in self._batch_metrics:
                batch_train_metrics[batch_metric.type][epoch] = res.batch_metrics[
                    batch_metric.type
                ]
            for epoch_metric in self._epoch_metrics:
                epoch_train_metrics[epoch_metric.type][epoch] = res.epoch_metrics[
                    epoch_metric.type
                ]

        for epoch, res in self._epoch_validation_results.items():
            batch_validation_losses[epoch] = res.batch_losses
            for batch_metric in self._batch_metrics:
                batch_validation_metrics[batch_metric.type][epoch] = res.batch_metrics[
                    batch_metric.type
                ]
            for epoch_metric in self._epoch_metrics:
                epoch_validation_metrics[epoch_metric.type][epoch] = res.epoch_metrics[
                    epoch_metric.type
                ]

        num_epochs = self._current_epoch
        self.fit_result = FitResult(
            batch_train_losses,
            batch_validation_losses,
            batch_train_metrics,
            batch_validation_metrics,
            epoch_train_metrics,
            epoch_validation_metrics,
            num_epochs,
            model,
        )
        self._log_fit(logs, fit_result=self.fit_result)
        return self.fit_result

    def _log_batch(self, logs: Optional[Dict], batch_result: BatchResult):
        """
        Logs the runtime and results of a single batch training. As full logging is
        implementation and environment dependent and should be implemented in
        inheriting classes.

        Args:
            logs: Additional logs collected during batch training
            batch_result: A BatchResult containing the batch loss and all metrics
        """
        pass

    def _log_epoch(self, logs: Optional[Dict], epoch_result: EpochResult):
        """
        Logs the runtime and results of a single epoch training. As full logging is
        implementation and environment dependent and should be implemented in
        inheriting classes.

        Args:
            logs: Additional logs collected during epoch training
            epoch_result: An EpochResult containing the losses and metrics for all
            batches in the epoch and aggregated epoch metrics.
        """
        pass

    def _log_fit(self, logs: Optional[Dict], fit_result: FitResult):
        """
        Logs the runtime and results of the entire model fit. As full logging is
        implementation and environment dependent and should be implemented in
        inheriting classes.

        Args:
            logs: Additional logs collected during the model fit
            fit_result: A FitResult containing the losses and metrics for all
            batches in all epochs and aggregated epoch metrics, as well as the
            trained model itself.
        """
        pass

    def _save_output(self, y: Dict[LabelType, Tensor], y_pred: Dict[LabelType, Tensor]):
        """
        Saves the output model predictions and their corresponding labels

        ** Not supported in the base class, inherit and implement if needed. **

        Args:
            y: The labels
            y_pred: The predictions

        """
        raise NotImplementedError()

    @property
    def current_epoch(self):
        """
        Returns: The number of the current epoch
        """
        return self._current_epoch
