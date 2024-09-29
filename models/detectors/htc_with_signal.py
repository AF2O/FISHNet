# Copyright (c) OpenMMLab. All rights reserved.
# from ..builder import DETECTORS
# from .cascade_rcnn import CascadeRCNN
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN


@DETECTORS.register_module()
class HybridTaskSignalCascade(CascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(HybridTaskSignalCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_signal_points=None,
                      gt_signal_labels=None,
                      gt_signal_ignore_area=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices codorresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # print('################# img shape ##################', img.size())
        import time
        start_time = time.time()
        x = self.extract_feat(img)
        print('extract_feat time', time.time() - start_time)
        losses = dict()

        # RPN forward and loss
        rpn_start_time = time.time()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        print('RPN forward time', time.time() - rpn_start_time)

        roi_head_start_time = time.time()
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,gt_bboxes_ignore,
                                                 gt_masks, gt_signal_points, gt_signal_labels,
                                                 gt_signal_ignore_area, **kwargs)
        print('roi_head time', time.time() - roi_head_start_time)
        losses.update(roi_losses)

        return losses


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        test_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        # print('############### test_results ###############', test_results)

        return test_results