from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from physlearn.apis.data import Sample, Signal
from physlearn.names import (
    DataBaseName,
    DataBaseVersion,
    DataType,
    LabelType,
    NonSignalDataType,
    SignalType,
)


class DataSource(ABC):
    """A base class for data sources.

    Allows the downstream components to load samples from different real sources in the
    same way, provides details about the data source and a signature.

    It assumes the following hierarchy of samples:
    - Source:
        - Patient 1:
            - Record 1:
                - Sample 1
                - Sample 2
                - ...
            - Record 2:
                - Sample 1
                - Sample 2
                - ...
            - ...
        - Patient 2:
            - Record 1:
                - Sample 1
                - Sample 2
                - ...
            - Record 2:
                - Sample 1
                - Sample 2
                - ...
            - ...
    This allows samples to be attributed to a record which correspond to a specific
    hospital visit in the real world.

    Note:
        The term sample may mean different things in specific kinds of data, always
        refer to the docs of the concrete source to understand the meaning in the
        specific context.
    """

    @abstractmethod
    def __getitem__(self, idx: Sequence[int]) -> Sample:
        """
        Returns a single sample from the data source. Uses a 3 integer index,
        corresponding to: `(patient ID, record ID, sample ID)`.

        Args:
            idx: Index of the sample from the entire data source, with the shape
            `(patient ID, record ID, sample ID)`.
        """
        pass

    @abstractmethod
    def record_ids_per_patient(self, patient_id: int) -> Sequence[int]:
        """
        Returns:
            Sequence of record IDs for the required patient
        """
        pass

    @abstractmethod
    def sample_ids_per_record(self, patient_id: int, record_id: int) -> Sequence[int]:
        """

        Args:
            patient_id: The id of the patient
            record_id: The id of the record

        Returns:
            Sequence of sample IDs in the requested record

        """
        pass

    @abstractmethod
    def _get_sample(
        self,
        patient_id: int,
        record_id: int,
        sample_id: int,
        index_dict: Dict[SignalType, Tuple],
    ) -> Tuple[Dict[SignalType, Signal], timedelta, Dict[NonSignalDataType, Any]]:
        """
        Extracts sample data for given patient ID, record ID, and sample indices.

        Args:
          patient_id: desired patient ID
          record_id: desired record ID
          index_dict: dictionary indicating (sample_start_idx, sample_len) for each
          signal type, when:
                sample_start_idx- index indicating starting of sample in the
                entire record
                sample_len- length of the sample

        Returns:
            signals: a dictionary of signal objects for each signal type found
        in the record.
            time: The time from the beginning of the record until the end of the
        sample.
            metadata: containing information such about the record such as seizure
        data, age, gender, atc.

        """
        pass

    @abstractmethod
    def __len__(self):
        """Total number of samples from all patients."""
        pass

    @property
    @abstractmethod
    def number_of_patients(self) -> int:
        """Number of patients in the data source"""
        pass

    @property
    @abstractmethod
    def samples_per_patient(self) -> Dict[int, int]:
        """A dictionary indicating the number of samples per patient"""
        pass

    @property
    @abstractmethod
    def name(self) -> DataBaseName:
        """The name of the source (e.g. hospital/database name)"""
        pass

    @property
    @abstractmethod
    def database_version(self) -> DataBaseVersion:
        """Database version of this data source"""
        pass

    @property
    @abstractmethod
    def properties(self) -> Any:
        """Properties of the current source, e.g. processing done, limitations."""
        pass

    @property
    @abstractmethod
    def signature(self) -> str:
        """The signature of the data source."""
        pass

    @property
    @abstractmethod
    def feature_types(self) -> Sequence[NonSignalDataType]:
        """Feature types in this data source"""
        pass

    @property
    @abstractmethod
    def signal_types(self) -> Sequence[SignalType]:
        """Signal types in this data source"""
        pass

    @property
    @abstractmethod
    def labels(self) -> Sequence[LabelType]:
        """Labels types in this data source"""

    @abstractmethod
    def _sign(self) -> str:
        """Calculates the signature of a data source.

        This will be done differently based on the way the data is saved. For data saved
        locally, hashing the files is the default.

        Returns: An hexadecimal digest of the hash.

        Notes:
            As signing of large databases may be long, the DataSource class should
            either do it lazily or to cache the signature.
        """
        pass

    @property
    @abstractmethod
    def patient_ids(self) -> Sequence[int]:
        """Sequence of valid patient IDs in the data source"""
        pass

    @property
    @abstractmethod
    def record_ids(self) -> Dict[int, Sequence[int]]:
        """Dictionary with records IDs per patient in the data source"""

    @property
    def sample_ids(self) -> Iterable[Sequence[int]]:
        """All the legal sample ids in this DataSource

        A concrete implementation of a generator that facilitates iteration over all
        "legal" indices in this data source.

        Yields: Legal indices

        """
        # for each patient
        for patient_id in self.patient_ids:
            # for each record
            for record_id in self.record_ids_per_patient(patient_id):
                # iterate over all the samples and yield their ids
                for sample_id in self.sample_ids_per_record(patient_id, record_id):
                    yield patient_id, record_id, sample_id


class DataSink(ABC):
    """
    Handles `Sample`s by streaming, saving, etc. Can serve as the sink of processing,
    loading or generation pipelines.

    Usage:
    >>> sink[sample.patient_id, sample.record_id, sample.sample_id] = sample
    """

    @abstractmethod
    def __setitem__(self, idx: Union[int, Sequence[int]], sample: Sample):
        """
        Gets a single sample and stores or handles it. Two indexing schemes are
        supported, indexing of the entire source as a single sequence or more fancy
        indexing by passing a sequence of indexes.
        Args:
            idx: Index to represent the provided Sample, can be a single
            integer for simple indexing of the entire source as one sequence or multiple
            integers to allow more fancy indexing (per patient, per record, etc.)
            sample: A sample to be handled by the sink.
        """
        pass


class Dataset(TorchDataset):
    """A base class for all datasets. Allows access to several data sources as a unified
    dataset. Implements pytorch Dataset API to allow DataLoader to load and batch
    samples efficiently.

    Dataset is expected to abstract all the complexity of handling multiple sources,
    varying length input and runtime processing and provide downstream components
    uniformly structured samples suitable for being an input for a torch model.

    """

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[DataType, Tensor], Dict[LabelType, Tensor]]:
        """
        Returns a single sample from the dataset. As expected by torch DataLoader, the
        indexing has to be simple consecutive integer scalars. All the complexity of
        complicated experiments has to be handled internally.

        Args:
            idx: Index of the sample from the entire dataset, including all sources, a
        single integer.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples from all data sources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the dataset (e.g. type of samples provided)"""
        pass

    @property
    @abstractmethod
    def signature(self) -> str:
        """The signature of the dataset."""
        pass
