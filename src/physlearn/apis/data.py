from datetime import timedelta
from typing import Any, Dict, Optional, Sequence

from physlearn.names import DataBaseName, LabelType, NonSignalDataType, SignalType
from physlearn.typing import Array


class Signal:
    """A signal base class.

    Args:
      start_time: The time from the beginning of the record until the beginning of the
    signal.
      end_time: The time from the beginning of the record until the beginning of the
    signal.
      time_axis: A tensor with the time axis of the signal, shape: [1, time steps].
      signal: The signal tensor of shape [channels, time steps].
      signal_type: What kind of signal is it?
      channel_names: sequence of channel names in the signal. For example: names of
    different EEG electrodes.

    """

    def __init__(
        self,
        start_time: timedelta,
        end_time: timedelta,
        time_axis: Array,
        signal: Array,
        signal_type: SignalType,
        channel_names: Optional[Sequence[str]] = None,
    ):
        """

        Args:
            start_time: The time from the beginning of the record until the beginning of
        the signal.
            end_time: The time from the beginning of the record until the beginning of
        the signal.
            time_axis: A tensor with the time axis of the signal,
        with shape: [1, time steps].
            signal: The signal tensor of shape [channels, time steps].
            signal_type: What kind of signal is it?
            channel_names: Names of the channels in the signal. For example: names of
        different EEG electrodes.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._time_axis = time_axis
        self._signal = signal
        self._signal_type = signal_type
        self._channel_names = channel_names
        self._num_channels = self.get_num_channels()
        self.check_dimensions()
        self.check_times()
        self._check_num_channels()

    def __getitem__(self, idx):
        """
        Syntactic sugar, allows getting data from a Signal object like:
        >>> signal[1:2, :1000]
        """
        return self.signal[idx].copy()

    @property
    def start_time(self):
        """ """
        return self._start_time

    @property
    def end_time(self):
        """ """
        return self._end_time

    @property
    def time_axis(self):
        """ """
        return self._time_axis.copy()

    @property
    def signal(self):
        """ """
        return self._signal.copy()

    @property
    def signal_type(self):
        """ """
        return self._signal_type

    @property
    def channel_names(self):
        """ """
        return self._channel_names

    @property
    def num_channels(self):
        """
        Number of channels in the signal.
        """
        return self._num_channels

    def get_num_channels(self):
        return self.signal.shape[0]

    def check_dimensions(self):
        """Signals can either be of the same length as the time axis or empty"""
        if not self.signal.size:
            return
        elif self.signal.shape[1] == self.time_axis.shape[1]:
            return
        else:
            raise ValueError(
                "Illegal signal dimensions, the signal has to be of the "
                "same length as the time axis or empty"
            )

    def check_times(self):
        """End time greater then start time"""
        if self.end_time <= self.start_time:
            raise ValueError("End time can't be before or same as start time")

    def __eq__(self, other):
        return bool(
            self.start_time == other.start_time
            and self.end_time == other.end_time
            and (self.time_axis == other.time_axis).all()
            and (self.signal == other.signal).all()
            and self.signal_type == other.signal_type
        )

    def _check_num_channels(self):
        """
        Checks number of channel match channel names
        """
        if self.channel_names:
            if self.num_channels != len(self.channel_names):
                raise ValueError("Number of channels mismatch channel names")

    def find_channel(self, desired_channel_name: str):
        """
        Search for a given channel name in the signal's channel names.
        Raises a value error if the channel was not found
        Returns the index of the given channel in the list of channel names
        of the signal.
        """
        if desired_channel_name not in self.channel_names:
            raise ValueError(
                f"Channel {desired_channel_name} not found" f"in channel names"
            )
        else:
            return tuple(
                i for i, e in enumerate(self.channel_names) if e == desired_channel_name
            )

    def signal_like_this(
        self,
        start_time: timedelta = None,
        end_time: timedelta = None,
        time_axis: Array = None,
        signal: Array = None,
        signal_type: SignalType = None,
        channel_names: Optional[Sequence[str]] = None,
    ):

        if not type(self) == Signal:
            raise NotImplementedError("Inheriting classes should override this method")

        start_time = start_time if start_time else self.start_time
        end_time = end_time if end_time else self.end_time
        time_axis = time_axis if time_axis is not None else self.time_axis
        signal = signal if signal is not None else self.signal
        signal_type = signal_type if signal_type else self.signal_type
        channel_names = channel_names if channel_names else self.channel_names

        return Signal(
            start_time=start_time,
            end_time=end_time,
            time_axis=time_axis,
            signal=signal,
            signal_type=signal_type,
            channel_names=channel_names,
        )


class Sample:
    """
    A single sample from a single patient from a single database.
    May include a full or a partial recording.
    """

    def __init__(
        self,
        db: DataBaseName,
        db_version: str,
        patient_id: int,
        record_id: int,
        sample_id: int,
        signals: Dict[SignalType, Signal],
        data: Dict[NonSignalDataType, Array],
        metadata: Dict[NonSignalDataType, Any],
        label: Dict[LabelType, Array],
    ):
        """
        Args:
            db:  Name of the database the sample is taken from
            db_version:  Version of the database the sample is taken from
            patient_id:  ID of the patient the sample is taken from in the database
            record_id: ID of the record within the patient
            sample_id:  ID of the sample within the record
            signals: Signal data of the sample as a dictionary with the structure:
                signals = {
                            signal_type1: signal1,
                            signal_type2: signal2,
                              ...
                              }
            data: Non-signal data as tensors (clinical, socioeconomic, etc.)
            metadata: Non-signal data (clinical, socioeconomic, etc.)
            label: The label of the sample for a specific supervised learning task

            Note:
                `data` and `metadata` may look alike but they are used for completely
                 different purposes. `data` contains tensors, ready to be used as
                 features for ML models while `metadata` is unstructured and used for
                 internal operations of the system like labeling.
        """
        self._db = db
        self._db_version = db_version
        self._patient_id = patient_id
        self._record_id = record_id
        self._sample_id = sample_id
        self._signals = signals
        self._data = data
        self._metadata = metadata
        self._label = label

    @property
    def record_id(self):
        return self._record_id

    def sample_like_this(
        self,
        db: Optional[DataBaseName] = None,
        db_version: Optional[str] = None,
        patient_id: Optional[int] = None,
        sample_id: Optional[int] = None,
        signals: Optional[Dict[SignalType, Signal]] = None,
        data: Optional[Dict[NonSignalDataType, Array]] = None,
        metadata: Optional[Dict[NonSignalDataType, Any]] = None,
        label: Optional[Dict[LabelType, Array]] = None,
        record_id: Optional[int] = None,
        **kwargs,
    ):
        if not type(self) == Sample:
            raise NotImplementedError("Inheriting classes should override this method")

        db = db if db else self.db
        db_version = db_version if db_version else self.db_version
        patient_id = patient_id if patient_id else self.patient_id
        sample_id = sample_id if sample_id else self.sample_id
        signals = signals if signals else self.signals
        data = data if data else self.data
        metadata = metadata if metadata else self.metadata
        label = label if label else self.label
        record_id = record_id if record_id else self.record_id

        return Sample(
            db=db,
            db_version=db_version,
            patient_id=patient_id,
            record_id=record_id,
            sample_id=sample_id,
            signals=signals,
            data=data,
            metadata=metadata,
            label=label,
        )

    @property
    def db(self):
        """ """
        return self._db

    @property
    def db_version(self):
        """ """
        return self._db_version

    @property
    def patient_id(self):
        """ """
        return self._patient_id

    @property
    def sample_id(self):
        """ """
        return self._sample_id

    @property
    def signals(self):
        """ """
        return self._signals

    @property
    def data(self):
        """ """
        return self._data

    @property
    def metadata(self):
        """ """
        return self._metadata

    @property
    def label(self):
        """ """
        return self._label

    def __eq__(self, other):
        return all(
            [
                self.db == other.db,
                self.db_version == other.db_version,
                self.patient_id == other.patient_id,
                self.record_id == other.record_id,
                self.sample_id == other.sample_id,
                self.time == other.time,
                self.signals == other.signals,
                compare_tensor_dict(self.data, other.data),
                compare_tensor_dict(self.label, other.label),
            ]
        )


def compare_tensor_dict(d1: Dict[Any, Array], d2: Dict[Any, Array]):
    """Compare two dictionaries of tensors

    Args:
      d1: First dictionary
      d2: Second  dictionary
    Returns:
        True if same, false otherwise.

    """
    if not d1 and not d2:
        return True

    return (
        all([(t1 == t2).all() for t1, t2 in zip(d1.values(), d2.values())])
        and d1.keys() == d2.keys()
    )
