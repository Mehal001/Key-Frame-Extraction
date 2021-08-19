from abc import ABCMeta, abstractmethod


class BaseAssigner_new(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, gt_labels_2=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
