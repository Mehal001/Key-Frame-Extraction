
import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps
import snoop
IS_MY_VERSION = True
# @snoop
def multiclass_nms(multi_bboxes,
                   multi_cls_scores,
                   multi_visib_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    # exit(0)
    num_obj_classes = multi_cls_scores.size(1) - 1
    if IS_MY_VERSION:
        num_visib_classes = multi_visib_scores.size(1) 
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_cls_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_cls_scores.size(0), num_obj_classes, 4)

    cls_scores = multi_cls_scores[:, :-1]
    if IS_MY_VERSION:
        visib_scores = multi_visib_scores

    labels = torch.arange(num_obj_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(cls_scores)

    if IS_MY_VERSION:
        labels_2 = torch.arange(num_visib_classes, dtype=torch.long) + 1
        labels_2 = labels_2.view(1, -1).expand_as(visib_scores)   
    bboxes = bboxes.reshape(-1, 4)
    cls_scores = cls_scores.reshape(-1)
    labels = labels.reshape(-1)
    if IS_MY_VERSION:
        visib_scores = visib_scores.reshape(-1)
        labels_2 = labels_2.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask_cls = cls_scores > score_thr
        if IS_MY_VERSION:
            valid_mask_visb = visib_scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_cls_scores.size(0), num_obj_classes)
        score_factors = score_factors.reshape(-1)
        cls_scores = cls_scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        cls_inds = valid_mask_cls.nonzero(as_tuple=False).squeeze(1)
        if IS_MY_VERSION:
            visib_inds = valid_mask_visb.nonzero(as_tuple=False).squeeze(1)
        if IS_MY_VERSION:
            bboxes, cls_scores, labels, visib_scores, labels_2 = bboxes[cls_inds], cls_scores[cls_inds], labels[cls_inds], visib_scores[visib_inds], labels_2[visib_inds]
        else:
            bboxes, cls_scores, labels = bboxes[cls_inds], cls_scores[cls_inds], labels[cls_inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        cls_scores = torch.cat([cls_scores, cls_scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
        if IS_MY_VERSION:
            visib_scores = torch.cat([visib_scores, visib_scores.new_zeros(1)], dim=0)
            labels_2 = torch.cat([labels_2, labels_2.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_inds:
            if IS_MY_VERSION:
                return bboxes, labels, labels_2, cls_inds
            else:
                return bboxes, labels, cls_inds
        else:
            if IS_MY_VERSION:
                return bboxes, labels
            else:                 
                return bboxes, labels

    dets_cls, keep_cls = batched_nms(bboxes, cls_scores, labels, nms_cfg)
    # if IS_MY_VERSION:
    #     dets_visib, keep_visib = batched_nms(bboxes_visib, visib_scores, labels_2, nms_cfg)

    if max_num > 0:
        dets = dets_cls[:max_num]
        # print(dets)
        keep_cls = keep_cls[:max_num]
        if IS_MY_VERSION:
            # exit(0)
            dets_visib = dets[:,:-1]
            bboxes_test = multi_bboxes.view(multi_cls_scores.size(0), -1, 4)
            inds =[]
            for i in range(0, len(bboxes_test)):
                for box in bboxes_test[i]:
                    for det in dets_visib:
                        if torch.equal(det,box):
                            inds.append(i)
            #gather 
            scr = torch.max(multi_visib_scores[inds], 1)
            visib_scr = scr[0]
            visib_scr = visib_scr.unsqueeze(1)
            visib_labels = scr[1]+1
            visib_labels = visib_labels.unsqueeze(1)
            visib_labels = visib_labels.type(torch.float32)
            dets = torch.cat([dets, visib_scr, visib_labels], dim=1)
            # print(dets)

    if return_inds:
        if IS_MY_VERSION:
            return dets, labels[keep_cls], keep_cls
        else:
            return dets, labels[keep_cls], keep_cls
    else:
        if IS_MY_VERSION:
            return dets, labels[keep_cls]
        else:
            return dets, labels[keep_cls]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
