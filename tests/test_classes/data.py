from datetime import timedelta
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
from numpy import linspace
from numpy.random import randint, randn

from physlearn.apis.data import Sample, Signal
from physlearn.apis.data_loading import DataSource
from physlearn.names import (
    DataBaseName,
    DataBaseVersion,
    LabelType,
    NonSignalDataType,
    SignalType,
)


class FakeDataSource(DataSource):
    def __init__(self):

        self._number_of_patients = randint(10, 20)
        self._patient_ids = tuple(list(range(self.number_of_patients)))
        self._num_records_per_patient = randint(2, 20, self.number_of_patients)
        self._num_samples_per_record = [
            randint(1, 5, r) for r in self._num_records_per_patient
        ]
        self._record_ids = {p: self.record_ids_per_patient(p) for p in self.patient_ids}

        self._source_dict = {}
        self.len = 0
        for patient_id in range(self._number_of_patients):
            for record_id in self.record_ids_per_patient(patient_id):
                for sample_id in self.sample_ids_per_record(patient_id, record_id):
                    t = randint(0, 1000)
                    dt = randint(1, 10)
                    sig = Signal(
                        start_time=timedelta(t),
                        end_time=timedelta(t + dt),
                        signal=randn(1, 200),
                        signal_type=SignalType.FAKE,
                        time_axis=np.expand_dims(linspace(t, t + dt, 200), axis=0),
                    )
                    samp = Sample(
                        self.name,
                        "fake version",
                        patient_id=patient_id,
                        record_id=record_id,
                        sample_id=sample_id,
                        signals={SignalType.FAKE: sig},
                        data={k: torch.rand(7) for k in self.feature_types},
                        metadata=None,
                        label={k: torch.rand(7) for k in self.labels},
                    )
                    self._source_dict[(patient_id, record_id, sample_id)] = samp
                    self.len += 1

    def __getitem__(self, idx: Union[int, Sequence[int]]) -> Sample:
        return self._source_dict[idx]

    def record_ids_per_patient(self, patient_id: int) -> Sequence[int]:
        return tuple(list(range(self._num_records_per_patient[patient_id])))

    def sample_ids_per_record(self, patient_id: int, record_id: int) -> Sequence[int]:
        return tuple(list(range(self._num_samples_per_record[patient_id][record_id])))

    def __len__(self):
        return self.len

    @property
    def number_of_patients(self) -> int:
        return self._number_of_patients

    @property
    def patient_ids(self) -> Sequence[int]:
        return self._patient_ids

    @property
    def samples_per_patient(self) -> Sequence[int]:
        pass

    @property
    def name(self) -> DataBaseName:
        return DataBaseName.FAKE_SOURCE

    @property
    def database_version(self) -> DataBaseVersion:
        return DataBaseVersion.FAKE

    @property
    def properties(self) -> Any:
        return None

    @property
    def signature(self) -> str:
        return "fake signature"

    @property
    def feature_types(self) -> Sequence[NonSignalDataType]:
        return (NonSignalDataType.FAKE,)

    @property
    def signal_types(self) -> Sequence[SignalType]:
        return (SignalType.FAKE,)

    @property
    def labels(self) -> Sequence[LabelType]:
        return (LabelType.FAKE,)

    @property
    def record_ids(self) -> Dict[int, Sequence[int]]:
        return self._record_ids

    def _sign(self) -> str:
        return "fake signature"

    def calc_sample_ids(self):
        sample_ids = []
        for patient in self.patient_ids:
            for record in range(len(self.record_ids[patient])):
                sample_ids.append()

    def _get_sample(
        self, patient_id, record_id, sample_start_idx, sample_len
    ) -> Tuple[Dict[SignalType, Signal], timedelta, Dict[NonSignalDataType, Any]]:
        signals = {SignalType.FAKE, None}
        time = timedelta(0)
        metadata = {NonSignalDataType, 0}
        return signals, time, metadata


class OtherFakeDataSource(FakeDataSource):
    @property
    def feature_types(self) -> Sequence[NonSignalDataType]:
        return (NonSignalDataType.FAKE, NonSignalDataType.NON_SIGNAL_DATA)

    @property
    def signature(self) -> str:
        return "other fake source"
