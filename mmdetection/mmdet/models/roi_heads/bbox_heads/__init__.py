from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead

from .bbox_head_new import BBoxHead_new
from .convfc_bbox_head_new import (ConvFCBBoxHead_new, Shared2FCBBoxHead_new,
                                  Shared4Conv1FCBBoxHead_new)

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead',
    'BBoxHead_new','ConvFCBBoxHead_new', 'Shared2FCBBoxHead_new', 'Shared4Conv1FCBBoxHead_new'
]
