import pytest

from physlearn.apis.data import Signal
from physlearn.names import SignalType
from tests.test_classes.data import FakeDataSource


@pytest.fixture
def sample():
    return FakeDataSource()[0, 0, 0]


@pytest.fixture
def example_signal(sample):
    return sample.signals[SignalType.FAKE]


class TestSignal:
    def test_data_safety(self, example_signal: Signal):
        with pytest.raises(AttributeError, match=r"can't set attribute"):
            example_signal.start_time = 17
        with pytest.raises(AttributeError, match=r"can't set attribute"):
            example_signal.start_time = 17
        with pytest.raises(AttributeError, match=r"can't set attribute"):
            example_signal.end_time = 17
        with pytest.raises(AttributeError, match=r"can't set attribute"):
            example_signal.time_axis = 17
        with pytest.raises(AttributeError, match=r"can't set attribute"):
            example_signal.signal = 17
        with pytest.raises(AttributeError, match=r"can't set attribute"):
            example_signal.signal_type = 17

    def test_data_validation(self, example_signal: Signal):
        with pytest.raises(ValueError, match=r"Illegal signal dimensions"):
            Signal(
                start_time=example_signal.start_time,
                end_time=example_signal.end_time,
                time_axis=example_signal.time_axis[:, 0:17],
                signal=example_signal.signal,
                signal_type=example_signal.signal_type,
            )
        with pytest.raises(ValueError, match=r"can't be before"):
            Signal(
                start_time=example_signal.end_time,
                end_time=example_signal.start_time,
                time_axis=example_signal.time_axis,
                signal=example_signal.signal,
                signal_type=example_signal.signal_type,
            )

    def test_getitem(self, example_signal):
        assert (example_signal.signal == example_signal[:, :]).all()

    def test_comparison(self, example_signal):
        equal_sig = Signal(
            start_time=example_signal.start_time,
            end_time=example_signal.end_time,
            time_axis=example_signal.time_axis,
            signal=example_signal.signal,
            signal_type=example_signal.signal_type,
        )

        different_signal = example_signal.signal
        different_signal[0, 17] = 17
        non_equal_sig = Signal(
            start_time=example_signal.start_time,
            end_time=example_signal.end_time,
            time_axis=example_signal.time_axis,
            signal=different_signal,
            signal_type=example_signal.signal_type,
        )

        assert example_signal == equal_sig
        assert example_signal != non_equal_sig


class TestSample:
    def test_sample_like(self, sample):
        assert sample == sample.sample_like_this()
        assert not sample == sample.sample_like_this(patient_id=17)
