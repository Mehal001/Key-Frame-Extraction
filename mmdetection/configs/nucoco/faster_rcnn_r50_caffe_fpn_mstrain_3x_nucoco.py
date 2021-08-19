# The new config inherits a base config to highlight the necessary modification
# from mmcv.runner import optimizer


_base_ = 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
roi_head=dict(
    bbox_head=dict(num_classes=6)))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('person', 'bicylce', 'car', 'motorcycle', 'bus', 'truck')
data = dict(
train=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/train/',
    classes=classes,
    ann_file='/home/students/acct1002_15/FYP/RRPN/data/nucoco/annotations/instances_train_6classes.json'),
val=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/val/',
    classes=classes,
    ann_file='/home/students/acct1002_15/FYP/RRPN/data/nucoco/annotations/instances_val_6classes.json'),
test=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/test/',
    classes=classes,
    ann_file='/home/students/acct1002_15/FYP/RRPN/data/nucoco/annotations/instances_test_6classes.json'))

# optimizer = dict( lr = 0.001)
#evaluation = dict(interval=100)
#total_epochs = 100
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# We can use the pre-trained Mask RCNN model to obtain higher performance

load_from = 'chkpoint/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'