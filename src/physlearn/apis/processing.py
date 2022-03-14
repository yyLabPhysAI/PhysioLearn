from abc import ABC
from typing import Dict

from epilepsy.apis.data import Sample, Signal
from epilepsy.names import LabelType, NonSignalDataType, SignalType
from torch import Tensor


class Processor(ABC):
    """
    A base class for sample processor - a component that apply pre-processing,
    normalization, data augmentation, etc.
    """

    def __init__(self):
        if type(self, Processor):
            raise TypeError(
                "Processor API shouldn't be instantiated directly. Use subclasses."
            )

    def __call__(self, x: Sample, **kwargs) -> Sample:
        """Process a sample

        Args:
            x: The input sample to process
            allow_mixing: If true, the input sample is saved as an instance attribute to
            allot access of all methods to all attributes of the input sample.
            **kwargs: For concrete implementations

        Returns:
            A processed sample

        """

        signals = self._process_signals(x.signals, data=x.data, label=x.label)
        data = self._process_data(x.data, signals=x.signals, label=x.label)
        features = self._extract_features(x.signals, data=x.data, label=x.label)
        label = self._process_label(
            x.label,
            signals=x.signals,
            data=x.data,
        )

        return x.sample_like_this(
            signals=signals, data={**data, **features}, label=label
        )

    def _process_signals(
        self, signals: Dict[SignalType, Signal], **kwargs
    ) -> Dict[SignalType, Signal]:
        """Processes signals

        Processes a dict of signals and returns a dict of processed signals.
        The processed signals may have a different time axis as some processing steps
        create a lag on the leading edge or require some history for computation. They
        also may be of a different type or have a different dict index.

        Args:
            signals: A dict of input signals
        """
        return signals

    def _process_data(
        self, data: Dict[NonSignalDataType, Tensor], **kwargs
    ) -> Dict[NonSignalDataType, Tensor]:
        """Processes data
        Processes a dict of data tensors and returns a dict of processed data.

        Args:
            data: Data to process
        """
        return data

    def _process_label(
        self, label: Dict[LabelType, Tensor], **kwargs
    ) -> Dict[LabelType, Tensor]:
        """Transforms labels

        Args:
            label: The original labels
        """
        return label

    def _extract_features(
        self, signals: Dict[SignalType, Signal], **kwargs
    ) -> Dict[NonSignalDataType, Tensor]:
        """Extracts features

        Extracts features from a collection of signals.

        Args:
            signals: A dict of input signals
        """
        return {}
