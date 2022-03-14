from abc import abstractmethod
from typing import Dict

from attr import attrib, attrs
from torch import Tensor

from physlearn.apis.data import Signal
from physlearn.names import SignalType


@attrs(frozen=True)
class RPeakSignal(Signal):
    """
    A signal of ECG R peaks
    indices: The indices of the R peaks in the original ECG time series. This tensor
    has to be at the same length as the time axis.
    The time axis in this case stands for the times of the R peaks.
    """

    signal = attrib(type=Tensor)

    @signal.validator
    def empty_signal(self, attribute, value):
        if value.shape[1]:
            raise ValueError(
                "R peak signal is a time axis of R wave times with an " "empty signal"
            )

    time_axis = attrib(type=Tensor)
    indices = attrib(type=Tensor)

    @indices.validator
    @time_axis.validator
    def check_dimensions(self, attribute, value):
        """
        Signals can either be of the same length as the time axis or empty
        """
        if self.indices.shape[1] == self.time_axis.shape[1]:
            return
        else:
            raise ValueError(
                "Illegal signal dimensions, the signal has to be of the "
                "same length as the time axis or empty"
            )


class RPeakDetector:
    """
    A base class for R peak detector.
    """

    def __call__(
        self, signals: Dict[SignalType, Signal], *args, **kwargs
    ) -> RPeakSignal:
        """
        Checks the signal type and calls the peak detection.
        Args:
            signals: ECG to extract R peaks from
            *args: Positional arguments of the peak detector
            **kwargs: Keyword arguments of the peak detector

        Returns: Signal of R peaks

        """
        if SignalType.ECG not in signals.keys():
            raise ValueError("RPeakDetectors need an ECG signal")
        return self._detect_peaks(signals, *kwargs, **kwargs)

    @abstractmethod
    def _detect_peaks(
        self, signals: Dict[SignalType, Signal], *args, **kwargs
    ) -> RPeakSignal:
        pass
