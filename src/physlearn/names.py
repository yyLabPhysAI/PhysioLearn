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


class NonSignalDataType(DataType):
    FAKE = "fake data type"
    CHANNEL_FREQUENCY = "channel_frequency"
    SIGNAL_NAME = "signal_name"
    SIGNAL_TYPE = "signal_type"
    EVENT_START_TIMES = "event start times"
    EVENT_END_TIMES = "event end times"
    POINT_ANNOTATION = "point annotation"
    HOSPITAL_NAME = "hospital_name"
    NUM_PATIENTS = "num_patients"
    NUM_RECORDS = "num_records"
    DATABASE_VERSION = "database_version"
    DATABASE_PROPERTIES = "database_properties"
    DATABASE_NAME = "database_name"
    PATIENT_ID = "patient_id"
    RECORDS_PER_PATIENT = "records_per_patient"
    NON_SIGNAL_DATA = "non_signal_data"
    RECORD_ID = "record_id"
    CHANNEL_FREQUENCY = "channel_frequency"


class LabelType(DataType):
    FAKE = "fake label"
    NO_LABEL = "no label"


class AnnotationType(DataType):
    POINT_ANNOTATION = "point annotation"
    INTERVAL_ANNOTATION = "interval annotation"


class HospitalName(Enum):
    FAKE = "fake"


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
