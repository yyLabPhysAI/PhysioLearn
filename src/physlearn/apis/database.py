from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence, Union

import numpy as np
from physiolearn.names import (
    AnnotationType,
    DataBaseName,
    DataBaseProperties,
    DataBaseVersion,
    HospitalName,
    NonSignalDataType,
    SignalType,
)

HDF_TYPES = Union[int, float, str, np.array, Sequence[Union[int, float, str, np.array]]]


class DatabaseMetadata:
    """
    Metadata included in a Database object.
    """

    def __init__(
        self,
        db_name: DataBaseName,
        db_properties: DataBaseProperties,
        db_version: DataBaseVersion,
        hospital: HospitalName,
        non_signal_data: Sequence[NonSignalDataType],
        signal_types: Sequence[SignalType],
    ):
        """
        Args:
            db_name: the name of the database
            db_properties: properties of the database, for example raw, preprocessed,
                etc.
            db_version: version of the database
            hospital: hospital name of the database
            non_signal_data: all non signal data types included in the database
            signal_types: all signal types included in the database
        """
        self.db_name = db_name
        self.db_properties = db_properties
        self.db_version = db_version
        self.hospital = hospital
        self.non_signal_data = non_signal_data
        self.signal_types = signal_types

    def __eq__(self, other):
        return all(
            [
                self.db_name == other.db_name,
                self.db_properties == other.db_properties,
                self.db_version == other.db_version,
                self.hospital == other.hospital,
                set(self.non_signal_data) == set(other.non_signal_data),
                set(self.signal_types) == set(other.signal_types),
            ]
        )


class RecordAnnotation(ABC):
    """
    This object contains the annotations of a record.
    """

    def __init__(self, patient_id: int, record_id: int):
        """
        Args:
            patient_id: patient ID of the record
            record_id: record ID
        """
        self.patient_id = patient_id
        self.record_id = record_id
        self.annotation = self.get_annotation()

    @abstractmethod
    def get_annotation(
        self,
    ) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Returns:
             annotation data
        """
        pass

    @abstractmethod
    def annotation_type(self) -> AnnotationType:
        """
        Returns:
            annotation type
        """
        pass

    @abstractmethod
    def _build_attrs_dict(
        self,
    ) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Builds an attributes dictionary that is compatible with HDF types and is used
        for saving in an HDF file.
        """
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


class Channel:
    """
    A channel object containing signal and its metadata from a single channel from a
     record.
    The signal itself is not included in this object for memory reasons. Instead,
    a callable object that can load the data by demand is given.
    """

    def __init__(
        self,
        get_channel: Callable[[], np.array],
        channel_id: int,
        signal_name: str,
        signal_type: SignalType,
        fs: int,
    ):
        """
        Args:
            get_channel: a function that loads the signal
            channel_id: id of the channel
            signal_name: the name of the channel
            signal_type: a SignalType object indicating the signal type of the channel
            fs: sampling frequency of the channel
        """
        self.get_channel = get_channel
        self.signal_name = signal_name
        self.signal_type = signal_type
        self.fs = fs
        self.channel_id = channel_id
        self._attrs_dict = self._build_attrs_dict()

    def _build_attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Builds an attributes dictionary that is compatible with HDF types and is used
        for saving in an HDF file.
        """
        attrs_dict = {
            NonSignalDataType.CHANNEL_FREQUENCY: self.fs,
            NonSignalDataType.SIGNAL_TYPE: self.signal_type.name,
            NonSignalDataType.SIGNAL_NAME: self.signal_name,
        }
        return attrs_dict

    @property
    def attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Attributes dictionary that can be used for saving in an HDF file.
        The values of the dictionary are stored in types that are compatible with HDF
        attributes.
        """
        return self._attrs_dict.copy()

    def __eq__(self, other):
        return all(
            [
                self.signal_name == other.signal_name,
                self.signal_type == other.signal_type,
                self.fs == other.fs,
                self.channel_id == other.channel_id,
                np.any(self.get_channel() == other.get_channel()),
            ]
        )


class RecordMetadata:
    """
    Metadata included in a Record object. Includes annotation data and additional data
    such as clinical information.
    """

    def __init__(
        self,
        patient_id: int,
        record_id: int,
        record_annotation: RecordAnnotation,
        metadata: Dict[NonSignalDataType, HDF_TYPES],
    ):
        """
        Args:
            patient_id: patient ID of the record
            record_id: record ID
            record_annotation: includes data regarding annotation of the record
            metadata: additional data regarding this record
        """
        self.patient_id = patient_id
        self.record_id = record_id
        self.annotation = record_annotation
        self.metadata = metadata
        self._attrs_dict = self._build_attrs_dict()

    def __eq__(self, other):
        return all(
            [
                self.patient_id == other.patient_id,
                self.record_id == other.record_id,
                self.annotation == other.annotation,
                self.metadata == other.metadata,
            ]
        )

    def _build_attrs_dict(
        self,
    ) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Builds an attributes dictionary that is compatible with HDF types and is used
        for saving in an HDF file.
        """
        attrs_dict = {**self.metadata, **self.annotation._build_attrs_dict()}
        return attrs_dict


class Record:
    """
    A record object containing signals and metadata from a single record.
    """

    def __init__(self, record_metadata: RecordMetadata, channels: Sequence[Channel]):
        """
        Args:
            record_metadata: metadata of the record
            channels: sequence of channel objects in the record
        """
        self.record_metadata = record_metadata
        self.channels = channels
        self._channel_ids = self._get_channel_ids()
        self._channel_dict = self._get_channel_dict()
        self._attrs_dict = self._build_attrs_dict()

    @property
    def attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Attributes dictionary that can be used for saving in an HDF file.
        The values of the dictionary are stored in types that are compatible with HDF
        attributes.
        """
        return self._attrs_dict.copy()

    def _build_attrs_dict(
        self,
    ) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Builds an attributes dictionary that is compatible with HDF types and is used
        for saving in an HDF file.
        """
        attrs_dict: Dict[
            NonSignalDataType,
            HDF_TYPES,
        ]
        attrs_dict = {
            NonSignalDataType.PATIENT_ID: self.record_metadata.patient_id,
            NonSignalDataType.RECORD_ID: self.record_metadata.record_id,
            **self.record_metadata._attrs_dict,
        }
        return attrs_dict

    def _get_channel_ids(self) -> Sequence[int]:
        """
        Returns channel IDs included in this record.
        """
        channel_ids = [ch.channel_id for ch in self.channels]
        return channel_ids

    @property
    def channel_ids(self) -> Sequence[int]:
        """
        A sequence of channel IDs included in this record.
        """
        return self._channel_ids

    def _get_channel_dict(self) -> Dict[int, Channel]:
        """
        Builds a dictionary containing channel IDs and the corresponding Channel
        objects included in this record.
        """
        channel_dict = {
            channel_id: channel
            for channel_id, channel in zip(self.channel_ids, self.channels)
        }
        return channel_dict

    @property
    def channel_dict(self) -> Dict[int, Channel]:
        """
        A dictionary containing channel IDs and the corresponding Channel
        objects included in this record.
        """
        return self._channel_dict

    def __eq__(self, other):
        channel_ids = self.channel_ids
        _, sorted_channels = zip(*sorted(zip(channel_ids, self.channels)))

        other_channel_ids = other.channel_ids
        _, other_sorted_channels = zip(*sorted(zip(other_channel_ids, other.channels)))

        if channel_ids != other_channel_ids:
            return False

        for ch, other_ch in zip(sorted_channels, other_sorted_channels):
            eq = np.array_equal(ch, other_ch)
            if not eq:
                return False

        return self.record_metadata == other.record_metadata


class Patient:
    """
    A patient object, containing records and metadata for a single patient.
    """

    def __init__(self, patient_id: int, records: Sequence[Record]):
        """
        Args:
            patient_id: id of the patient
            records: sequence of Record objects of the patient
        """
        self.patient_id = patient_id
        self.records = records
        self._record_ids = self.get_record_ids()
        self._record_dict = self.record_dict
        self._attrs_dict = self._build_attrs_dict()

    def get_record_ids(self) -> Sequence[int]:
        """
        Returns record IDs included in this patient.
        """
        record_ids = [record.record_metadata.record_id for record in self.records]
        return tuple(record_ids)

    @property
    def record_ids(self) -> Sequence[int]:
        """
        Record IDs included in this patient.
        """
        return self._record_ids

    @property
    def record_dict(self) -> Dict[int, Record]:
        """
        A dictionary indicating record IDs and the corresponding Record objects
        included in this patient.
        """
        record_dict = {
            record_id: record
            for record_id, record in zip(self.record_ids, self.records)
        }
        return record_dict

    @property
    def attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Attributes dictionary that can be used for saving in an HDF file.
        The values of the dictionary are stored in types that are compatible with HDF
        attributes.
        """
        return self._attrs_dict.copy()

    def _build_attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Builds an attributes dictionary that is compatible with HDF types and is used
        for saving in an HDF file.
        """
        record_ids = self.get_record_ids()
        attrs_dict: Dict[NonSignalDataType, HDF_TYPES]
        attrs_dict = {NonSignalDataType.RECORD_ID: record_ids}
        return attrs_dict

    def __eq__(self, other):
        record_ids = self.get_record_ids()
        _, sorted_records = zip(*sorted(zip(record_ids, self.records)))

        other_record_ids = other.get_record_ids()
        _, other_sorted_records = zip(*sorted(zip(other_record_ids, other.records)))

        if record_ids != other_record_ids:
            return False

        for r, other_r in zip(sorted_records, other_sorted_records):
            eq = r == other_r
            if not eq:
                return False

        return self.patient_id == other.patient_id


class Database:
    """
    A database object, containing patients data and metadata for a single database.
    The hierarchical order of the database is as follows:
    Hospital > Patient ID > Record ID > Signals from different channels.
    """

    def __init__(self, db_metadata: DatabaseMetadata, patients: Sequence[Patient]):
        """
        Args:
            db_metadata: metadata of the database
            patients: a sequence of Patient objects included in the database
        """
        self._db_metadata = db_metadata
        self.patients = patients
        self._num_patients = len(self.patients)
        self._num_records = self._get_num_records()
        self._patient_ids = self.get_patient_ids()
        self._records_per_patient = self._get_records_per_patient()
        self._patient_dict = self._get_patient_dict()
        self._attrs_dict = self._build_attrs_dict()

    def _get_num_records(self) -> int:
        """
        Returns the number of records in the database
        """
        num_records = sum(len(patient.get_record_ids()) for patient in self.patients)
        return num_records

    def get_patient_ids(self) -> Sequence[int]:
        """
        Returns patient IDs included in the database.
        """
        patient_ids = tuple([patient.patient_id for patient in self.patients])
        return patient_ids

    def _get_records_per_patient(self) -> Sequence[int]:
        """
        Returns a sequence indicating number of records per patient in the database.
        """
        records_per_patient = tuple(
            [len(patient.get_record_ids()) for patient in self.patients]
        )
        return records_per_patient

    def _get_patient_dict(self) -> Dict[int, Patient]:
        """
        Returns a dictionary indicating patient IDs and the corresponding Patient
        objects in the database.
        """
        patient_dict = {patient.patient_id: patient for patient in self.patients}
        return patient_dict

    @property
    def patient_dict(self) -> Dict[int, Patient]:
        """
        A dictionary indicating patient IDs and the corresponding Patient objects in
        the database.
        """
        return self._patient_dict

    @property
    def db_metadata(self) -> DatabaseMetadata:
        """
        Metadata of the database.
        """
        return self._db_metadata

    @property
    def num_patients(self) -> int:
        """
        Total number of patients in the database.
        """
        return self._num_patients

    @property
    def num_records(self) -> int:
        """
        Total number of records in the database.
        """
        return self._num_records

    @property
    def patient_ids(self) -> Sequence[int]:
        """
        A sequence of all patient IDs included in the database.
        """
        return self._patient_ids

    @property
    def records_per_patient(self) -> Sequence[int]:
        """
        A sequence indicating the number of records per patient, corresponding to
        patient IDs sequence.
        """
        return self._records_per_patient

    @property
    def attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Attributes dictionary that can be used for saving in an HDF file.
        The values of the dictionary are stored in types that are compatible with HDF
        attributes.
        """
        return self._attrs_dict.copy()

    def _build_attrs_dict(self) -> Dict[NonSignalDataType, HDF_TYPES]:
        """
        Builds an attributes dictionary that is compatible with HDF types and is used
        for saving in an HDF file.
        """
        attrs_dict = {
            NonSignalDataType.HOSPITAL_NAME: self.db_metadata.hospital.name,
            NonSignalDataType.NUM_PATIENTS: self.num_patients,
            NonSignalDataType.NUM_RECORDS: self.num_records,
            NonSignalDataType.SIGNAL_TYPE: [
                sig.name for sig in self.db_metadata.signal_types
            ],
            NonSignalDataType.DATABASE_VERSION: self.db_metadata.db_version.name,
            NonSignalDataType.DATABASE_PROPERTIES: self.db_metadata.db_properties.name,
            NonSignalDataType.DATABASE_NAME: self.db_metadata.db_name.name,
            NonSignalDataType.PATIENT_ID: self.patient_ids,
            NonSignalDataType.RECORDS_PER_PATIENT: self.records_per_patient,
            NonSignalDataType.NON_SIGNAL_DATA: [
                d.name for d in self.db_metadata.non_signal_data
            ],
        }
        return attrs_dict

    def __eq__(self, other):
        patient_ids = self.patient_ids
        _, sorted_patients = zip(*sorted(zip(patient_ids, self.patients)))

        other_patient_ids = other.patient_ids
        _, other_sorted_patients = zip(*sorted(zip(other_patient_ids, other.patients)))

        if patient_ids != other_patient_ids:
            return False

        for p, other_p in zip(sorted_patients, other_sorted_patients):
            eq = p == other_p
            if not eq:
                return False
        return self.db_metadata == other.db_metadata


def get_signal_types(patients: Sequence[Patient]) -> Sequence[SignalType]:
    """
    Get signal typed in the entire database.
    Args:
        patients: a sequence of Patient objects
    Returns:
        a sequence of signal types included in the all patients
    """
    signal_types = []
    for p in patients:
        for r in p.records:
            for ch in r.channels:
                if ch.signal_type not in signal_types:
                    signal_types.append(ch.signal_type)

    return signal_types


def get_non_signal_types(patients: Sequence[Patient]) -> Sequence[NonSignalDataType]:
    """
    Get non-data signal types included in the records given. All records are build with
    the same metadata API, therefore the first record indicates the non signal data
    included.
    Args:
        records: a sequence of records
    Returns:
        a sequence of non-signal data types included in all records.
    """
    non_signal_types = []
    for p in patients:
        for r in p.records:
            for d in r._attrs_dict.keys():
                if d not in non_signal_types:
                    non_signal_types.append(d)

    return non_signal_types
