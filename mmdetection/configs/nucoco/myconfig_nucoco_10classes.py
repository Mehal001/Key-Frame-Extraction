# The new config inherits a base config to highlight the necessary modification

_base_ = 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
roi_head=dict(
    bbox_head=dict(num_classes=10)))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('child', 'adult', 'bicycle', 'motorcycle', 'car', 'bus', 'truck', 'trailer', 'barrier', 'traffic_cone')
data = dict(
train=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/train/',
    classes=classes,
    ann_file='configs/nucoco/train/instances_train.json'),
val=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/val/',
    classes=classes,
    ann_file='configs/nucoco/val/instances_val.json'),
test=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/val/',
    classes=classes,
    ann_file='configs/nucoco/val/instances_val.json'))

# optimizer = dict( lr = 0.001)
#evaluation = dict(interval=100)
#total_epochs = 100
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# We can use the pre-trained Mask RCNN model to obtain higher performance
workflow = [('train', 1), ('val', 1)]
load_from = 'chkpoint/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'
