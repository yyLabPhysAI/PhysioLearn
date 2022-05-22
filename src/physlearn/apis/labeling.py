from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from torch import Tensor

from physlearn.apis.data import Sample
from physlearn.apis.metadata import LabelMetadata
from physlearn.names import LabelType, NonSignalDataType, SignalType


class Labeler(ABC):
    """A base class for sample labeler - a component that adds new labels to a sample based
    on metadata.
    """

    def __call__(self, x: Sample, metadata: LabelMetadata):
        """
        Labels a sample based on metadata.

        Args:
            x: A sample to label
            metadata: metadata to label upon

        Returns:
            A sample with additional labels based on the provided metadata

        """
        new_label = self.create_label(x, metadata)
        return x.sample_like_this(label={**x.label, **new_label})

    @abstractmethod
    def create_label(
        self, x: Sample, metadata: LabelMetadata
    ) -> Dict[LabelType, Dict[SignalType, Tensor]]:
        """
        Creates a label for each signal type in the sample given.

        Args:
          x: Sample
          metadata: LabelMetadata

        Returns:
            label: dictionary of label types and the labeling.
        A label tensor is created for each signal type in the sample.
        """
        pass

    @property
    @abstractmethod
    def label_type(self) -> Sequence[LabelType]:
        """
        Returns:
             label type of labeler.
        """
        pass


class MetadataExtractor(ABC):
    """A base class for the extraction of raw metadata coming and transforming it
    into a class designated to signal labeling.
    """

    @abstractmethod
    def build_metadata(
        self, record_metadata: Dict[NonSignalDataType, Any]
    ) -> LabelMetadata:
        """Extracts data from a given metadata, performs manipulations on it and
        returns a LabelMetadata object which contains all data necessary for
        labeling.

        Args:
          record_metadata: dictionary containing raw metadata of a record.

        Returns:
            a LabelMetadata object which contains all the data needed for labeling.

        """
        pass
