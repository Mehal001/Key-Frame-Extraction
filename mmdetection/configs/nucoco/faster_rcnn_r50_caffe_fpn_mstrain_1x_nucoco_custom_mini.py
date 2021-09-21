# The new config inherits a base config to highlight the necessary modification
# from mmcv.runner import optimizer


_base_ = 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco_custom.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
roi_head=dict(
    bbox_head=dict(num_object_classes=10)))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('child', 'adult', 'bicycle', 'motorcycle', 'car', 'bus', 'truck', 'trailer', 'barrier', 'traffic_cone')
data = dict(
train=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/mini_train/',
    classes=classes,
    ann_file='configs/nucoco/mini_train/instances_mini_train_int_visib.json'),
val=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/mini_val/',
    classes=classes,
    ann_file='configs/nucoco/mini_val/instances_mini_val_int_visib.json'),
test=dict(
    type = dataset_type,
    img_prefix='configs/nucoco/mini_val/',
    classes=classes,
    ann_file='configs/nucoco/mini_val/instances_mini_val_int_visib.json'))

optimizer = dict( lr = 0.005)
workflow = [('train', 1), ('val', 1)]
#evaluation = dict(interval=100)
#total_epochs = 100
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'chkpoint/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'