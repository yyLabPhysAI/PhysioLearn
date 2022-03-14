from abc import ABC, abstractmethod
from typing import Dict, Sequence

import torch
from torch import Tensor

from physlearn.apis.training.results import BatchResult
from physlearn.names import BatchMetricType, EpochMetricType, LabelType


class BatchMetric(ABC):
    """
    A base class for metrics calculated over a single batch
    """

    def __call__(
        self, y: Dict[LabelType, Tensor], y_pred: Dict[LabelType, Tensor]
    ) -> float:
        with torch.no_grad():
            return self._metric(y, y_pred)

    @abstractmethod
    def _metric(
        self, y: Dict[LabelType, Tensor], y_pred: Dict[LabelType, Tensor]
    ) -> float:
        """
        Implements tha metric itself in concrete classes.

        Args:
            y: Batch labels
            y_pred: Batch model predictions

        Returns: The metric as a simple float

        """
        pass

    @property
    @abstractmethod
    def type(self) -> BatchMetricType:
        """
        Returns: The type of the metric
        """
        pass


class EpochMetric:
    """
    A base class for metrics aggregated from all batch results over a single epoch
    """

    def __call__(self, batch_results: Sequence[BatchResult]) -> float:
        for metric_type in self.required_batch_metrics:
            for batch in batch_results:
                if metric_type not in batch.metrics.keys():
                    raise IndexError(
                        f"Metric {metric_type.value} is missing in a batch"
                    )
        return self._metric(batch_results)

    @abstractmethod
    def _metric(self, batch_results: Sequence[BatchResult]) -> float:
        """
        Implements tha metric itself in concrete classes.

        Args:
            batch_results: Batch results to aggregate from

        Returns: The aggregated metric as a simple float
        """
        pass

    @property
    @abstractmethod
    def type(self) -> EpochMetricType:
        """
        Returns: The type of the metric
        """
        pass

    @property
    @abstractmethod
    def required_batch_metrics(self) -> Sequence[BatchMetricType]:
        """
        Returns: The batch metrics required for this epoch metric.
        """
        pass
