from enum import Enum


class DataType(Enum):
    def __lt__(self, other):
        return self.value <= other.value

    def __le__(self, other):
        return self.value < other.value


class ModelDataType(DataType):
    FEATURES = "features"


class SignalType(DataType):
    ECG = "ecg"
    EEG = "eeg"
    RRIntervals = "rri"
    FAKE = "fake signal"
    Unknown = "unknown"
    SPO2 = "spo2"
    HR = "hr"


class NonSignalDataType(DataType):
    HOSPITAL_NAME = "hospital_name"
    AGE = "age"
    GENDER = "gender"
    FAKE = "fake data type"
    RECORD_SAMPLING_FREQUENCY = "record_sampling_frequency"
    RECORD_FREQUENCY = "record_frequency"
    CHANNEL_FREQUENCY = "channel_frequency"
    SIGNAL_NAME = "signal_name"
    NON_SIGNAL_DATA = "non_signal_data"
    PATIENT_IDS = "patient_ids"
    PATIENT_ID = "patient_id"
    RECORD_ID = "record_id"
    RECORDS_PER_PATIENT = "records_per_patient"
    SIGNAL_TYPES = "signal_types"
    TOTAL_NUMBER_OF_SEIZURES = "total_number_of_seizures"
    DATABASE_VERSION = "database_version"
    DATABASE_PROPERTIES = "database_properties"
    NUM_RECORDS = "num_records"
    NUM_PATIENTS = "num_patients"
    DATABASE_NAME = "database_name"
    START_DATE = "start_date"
    START_TIME = "start_time"
    END_DATE = "end_date"
    END_TIME = "end_time"
    SIGNAL_DURATION_SEC = "signal_duration_sec"
    TIME_AXIS = "time_axis"
    RECORD_IDS = "record_ids"
    SIGNAL_LEN = "signal_len"
    SAMPLE_IDX = "sample_idx"


class LabelType(DataType):
    FAKE = "fake label"
    NO_LABEL = "no label"


class HospitalName(Enum):
    pass


class DataBaseVersion(Enum):
    V0 = "v0"
    FAKE = "fake version"


class DataBaseProperties(Enum):
    RAW = "raw"
    FOR_UNIT_TEST = "for_unit_test"


class DataBaseName(Enum):
    FAKE_SOURCE = "fake_source"


class BatchMetricType(Enum):
    ACCURACY = "accuracy"
    NUM_CORRECT = "Number of correct classifications"
    TRUE_NEGATIVE = "Number of correct negative classifications"
    TRUE_POSITIVE = "Number of correct positive classifications"
    FALSE_NEGATIVE = "Number of false negative classifications"
    FALSE_POSITIVE = "Number of false positive classifications"
    PPV = "Positive predictive value"
    SENSITIVITY = "Sensitivity"
    SPECIFICITY = "Specificity"
    F1 = "F-score"
    AUROC = "Area under ROC curve"


class EpochMetricType(Enum):
    ACCURACY = "accuracy"
    PPV = "Positive predictive value"
    SENSITIVITY = "Sensitivity"
    SPECIFICITY = "Specificity"
    F1 = "F-score"
    AUROC = "Area under ROC curve"


class LossType(Enum):
    MSE = "Mean square error"
    BCE = "Binary cross entropy"
    CROSS_ENTROPY = "Cross entropy"
    RAW = "raw"


class CallbackType(Enum):
    PLOT_TRAINING_LOSS = "Plot the training loss"
    CHECKPOINTS = "Save checkpoints when a metric improve"
    LR_SCHEDULER = "Learning rate scheduler"


class StopperType(Enum):
    MAX_EPOCHS = "Maximal number of epochs has been reached"
    EARLY_STOPPING = "No learning progression"
