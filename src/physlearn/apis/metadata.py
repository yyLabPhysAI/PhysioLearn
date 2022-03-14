from abc import ABC, abstractmethod

from physlearn.names import DataBaseName, LabelType


class LabelMetadata(ABC):
    """
    A class defining the metadata required for a labeler.
    """

    @property
    @abstractmethod
    def db_name(self) -> DataBaseName:
        """
        Returns:
             data base name the record was taken from.
        """
        pass

    @property
    @abstractmethod
    def patient_id(self) -> int:
        """
        Returns:
             patient ID of the record.
        """
        pass

    @property
    @abstractmethod
    def record_id(self) -> int:
        """
        Returns:
             record ID of the record.
        """
        pass

    @property
    @abstractmethod
    def label_type(self) -> LabelType:
        """
        Returns:
             the label type of the destined labeler.
        """
        pass
