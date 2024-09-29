# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.utils.brick_wrappers import adaptive_avg_pool2d

from .cascade_fish_roi_head import CascadeSignalRoIHead
from Swim_Fish.core.post_processing.signal_aug import get_aug_signals
from Swim_Fish.utils.signal_head_utils import vis_points, vis_match_points


@HEADS.register_module()
class HTCFishRoIUNetHead(CascadeSignalRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 signal_roi_extractor=None,
                 signal_head=None,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 signal_info_flow=True,
                 **kwargs):
        super(HTCFishRoIUNetHead, self).__init__(
            num_stages = num_stages, stage_loss_weights = stage_loss_weights,
            bbox_roi_extractor = bbox_roi_extractor, bbox_head = bbox_head,
            mask_roi_extractor = mask_roi_extractor, mask_head = mask_head,
            signal_roi_extractor = signal_roi_extractor, signal_head = signal_head, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.signal_info_flow = signal_info_flow

        if signal_head is not None:
            print('################# self.signal_roi_extractor #################', self.signal_roi_extractor)
            # print('################# self.signal_head #################', self.signal_head)
            # print('################# self.mask_head #################', self.mask_head)


    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    @property
    def with_signal(self):
        return True

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage, x, rois, semantic_feat=semantic_feat)

        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                             gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'], rois,
                                   *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            bbox_targets=bbox_targets,
        )
        return bbox_results



    def set_anchor(self, num_rois, mask_feature_size, stride = 1):

        shift_x = (np.arange(0, mask_feature_size, stride))
        shift_y = (np.arange(0, mask_feature_size, stride))

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        K = shifts.shape[0]
        all_anchor_points = shifts.reshape((1, K, 2))
        all_anchor_points = all_anchor_points.repeat(num_rois, 0)
        # print('all_anchor_points', all_anchor_points, len(all_anchor_points[0]))
        rel_roi_anchor_points = torch.tensor(all_anchor_points / mask_feature_size)


        return rel_roi_anchor_points

    def get_points_in_roi_small_image(self, rois, points_in_roi_small_image):


        xs = points_in_roi_small_image[:, :, 0] * (rois[:, None, 3] - rois[:, None, 1])
        ys = points_in_roi_small_image[:, :, 1] * (rois[:, None, 4] - rois[:, None, 2])

        pred_coords = torch.stack([xs, ys], dim=2)

        return pred_coords

    def get_abs_points(self, rois, points_in_rel_roi):
        # print('#################### rois ###########', rois[0])
        xs = points_in_rel_roi[:, :, 0] * (rois[:, None, 3] - rois[:, None, 1]) + rois[:, None, 1]
        ys = points_in_rel_roi[:, :, 1] * (rois[:, None, 4] - rois[:, None, 2]) + rois[:, None, 2]
        pred_abs_points = torch.stack([xs, ys], dim=2)
        return pred_abs_points

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        mask_roi_extractor = self.mask_roi_extractor[stage]

        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # print('************* rois see mask *************', pos_rois)
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](mask_feats, last_feat, return_logits=False)
            mask_pred, last_feat = mask_head(mask_feats, last_feat)
        else:
            mask_pred, last_feat = mask_head(mask_feats)

        mask_targets = mask_head.get_targets(sampling_results, gt_masks, rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)

        mask_results = dict(mask_pred=mask_pred, mask_last_feats=last_feat, loss_mask=loss_mask)

        return mask_results


    def _signal_forward_train(self, stage, x, sampling_results, mask_last_feat, gt_signal_points, gt_signal_labels):
        """Run forward function and calculate loss for siganl in training."""
        cur_level_signal_head = self.signal_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        print('stage', stage)

        num_rois = len(pos_rois)
        x = [x[i] for i in range(len(x)-1, -1, -1)][2:]
        print(x[0].size(), x[1].size(), x[2].size())

        # forward all previous signal heads to obtain last_feat, and fuse it
        # with the normal signal feature
        if self.signal_info_flow:
            last_feat = None
            for i in range(stage):
                fusion_feature = self.signal_head[i].forward_fusion(x, self.signal_roi_extractor, pos_rois, mask_last_feat, last_feat)
                last_feat = self.signal_head[i](fusion_feature, reture_conv3_3 = True)
                print('----------------- i --------------', i)
            fusion_feature = cur_level_signal_head.forward_fusion(x, self.signal_roi_extractor, pos_rois, mask_last_feat, last_feat)
            signal_feature = cur_level_signal_head(fusion_feature)
        else:
            fusion_feature = cur_level_signal_head.forward_fusion(x, self.signal_roi_extractor, pos_rois, mask_last_feat, last_feat = None)
            signal_feature = cur_level_signal_head(fusion_feature)

        # print('signal_feature size', signal_feature.size())
        deltas, pred_logits = cur_level_signal_head.forward_signal(signal_feature)
        print('deltas, pred_logits', deltas.size(), pred_logits.size())
        rel_roi_anchor_points = self.set_anchor(num_rois, mask_feature_size = 14)
        # print('rel_roi_anchor_points', rel_roi_anchor_points.size())
        rel_roi_anchor_points = rel_roi_anchor_points.cuda(deltas.device)
        points_in_roi_small_image = deltas * 0.1 + rel_roi_anchor_points

        pred_coords = self.get_points_in_roi_small_image(pos_rois, points_in_roi_small_image)
        # vis_points(pred_coords, pred_logits, pos_rois)
        target_signals = cur_level_signal_head.get_targets(sampling_results, gt_signal_points, gt_signal_labels)

        tgt_points = []
        tgt_labels = []

        for target_signal in target_signals:
            tgt_points.extend(target_signal["points"])
            tgt_labels.extend(target_signal["labels"])

        indices = cur_level_signal_head.matcher.matcher(pred_coords.cpu(), pred_logits.cpu(), tgt_points, tgt_labels)
        loss_signal = cur_level_signal_head.loss(pred_coords, pred_logits, tgt_points, tgt_labels, indices)

        signal_results = dict(signal_pred_coords=pred_coords, signal_pred_logits=pred_logits, loss_signal=loss_signal)

        return signal_results


    def get_pred_abs_points(self, rois, src_points):
        src_abs_points = []
        for each_src_points, each_roi in zip(src_points, rois):
            if len(each_src_points) > 0:
                xs = each_src_points[:, 0] + each_roi[1]
                ys = each_src_points[:, 1] + each_roi[2]
                coords = torch.stack([xs, ys], dim=1)
                src_abs_points.append(coords.detach().cpu().numpy())
            else:
                src_abs_points.append([])

        return src_abs_points

    def get_target_abs_points(self, rois, target_points):
        target_abs_points = []
        for each_tgt_points, each_roi in zip(target_points, rois):
            if len(each_tgt_points) > 0:
                xs = each_tgt_points[:, 0] + each_roi[1]
                ys = each_tgt_points[:, 1] + each_roi[2]
                coords = torch.stack([xs, ys], dim=1)
                target_abs_points.append(coords.detach().cpu().numpy())
            else:
                target_abs_points.append([])

        return target_abs_points

    def _bbox_forward(self, stage, x, rois, semantic_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        # print('############ bbox x ############', len(x), x[0].size())
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        # print('############ bbox_feats ############', len(bbox_feats), bbox_feats[0].size())
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_signal_points=None,
                      gt_signal_labels=None,
                      gt_signal_ignore_area=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features

        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg

        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            ### assign_each_proposal to each ground truth
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])

                            sampling_results.append(sampling_result)
                mask_results = self._mask_forward_train(i, x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat)
                signal_results = self._signal_forward_train(i, x, sampling_results, mask_results['mask_last_feats'],
                                                         gt_signal_points, gt_signal_labels)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

                for name, value in signal_results['loss_signal'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_signal_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                signal_reuslts = None
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                aug_signals = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat

                last_mask_feat = None
                last_signal_feat = None
                x_signal = [x[i] for i in range(len(x)-1, -1, -1)][2:]

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                print('num_stages', self.num_stages)
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_mask_feat = mask_head(mask_feats, last_mask_feat)
                    else:
                        mask_pred, last_mask_feat = mask_head(mask_feats)

                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append(
                        [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                    signal_head = self.signal_head[i]
                    if self.signal_info_flow:
                        fusion_feature = signal_head.forward_fusion(x_signal, self.signal_roi_extractor, mask_rois,
                                                                            last_mask_feat, last_signal_feat)
                        if i == 2:
                            reture_conv3_3 = False
                        else:
                            reture_conv3_3 = True
                        last_signal_feat = signal_head(fusion_feature, reture_conv3_3=reture_conv3_3)
                        signal_feature = last_signal_feat
                        print('last_signal_feat size', last_signal_feat.size())

                    else:
                        fusion_feature = signal_head.forward_fusion(x_signal, self.signal_roi_extractor, mask_rois,
                                                                              last_mask_feat)
                        signal_feature = signal_head(fusion_feature)

                deltas, pred_logits = signal_head.forward_signal(signal_feature)
                print('deltas pred_logits', deltas.size(), pred_logits.size())

                num_rois = len(mask_rois)
                rel_roi_anchor_points = self.set_anchor(num_rois, mask_feature_size=14)
                rel_roi_anchor_points = rel_roi_anchor_points.cuda(deltas.device)
                points_in_roi_small_image = deltas * 0.1 + rel_roi_anchor_points

                pred_abs_points = self.get_abs_points(mask_rois, points_in_roi_small_image)
                pred_logits = torch.argmax(pred_logits, dim=-1, keepdim=True)

                # pred_abs_points, pred_logits = self._signal_forward_test(x, mask_rois, mask_feats)
                pred_abs_points = pred_abs_points.split(num_bbox_per_img, 0)
                pred_logits = pred_logits.split(num_bbox_per_img, 0)
                aug_signals.append([torch.cat([signal_point, signal_logit], dim=-1).cpu() for
                                    signal_point, signal_logit in zip(pred_abs_points, pred_logits)])

                # apply mask post-processing to each image individually
                segm_results = []
                signal_reuslts = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        aug_signal = aug_signals[i]
                        merged_mask = merge_aug_masks(aug_mask, [[img_metas[i]]] * self.num_stages, rcnn_test_cfg)
                        sig_result = get_aug_signals(aug_signal, [img_metas[i]])
                        segm_result = self.mask_head[-1].get_seg_masks(merged_mask, _bboxes[i], det_labels[i],
                                                                       rcnn_test_cfg, ori_shapes[i], scale_factors[i], rescale)
                        segm_results.append(segm_result)
                        signal_reuslts.append(sig_result)
            ms_segm_result['ensemble'] = segm_results
            ms_signal_result['ensemble'] = signal_reuslts
            # print('########### ensemble_signal_results ##############', len(signal_reuslts), len(signal_reuslts[0]), signal_reuslts[0][0].shape)

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], ms_signal_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1] for feat in img_feats
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(img_feats, img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas,
                                                 semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result, _ = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    None,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]
