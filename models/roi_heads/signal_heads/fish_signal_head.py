# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import HEADS, build_loss
# from mmcv.runner import BaseModule
from scipy.optimize import linear_sum_assignment
from torch import nn

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead

from Swim_Fish.utils.signal_head_utils import *
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.double_conv(x)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)

        return out


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(RegressionModel, self).__init__()

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out

class MaskConv(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(MaskConv, self).__init__()
        self.conv1 = DoubleConv(num_features_in, feature_size)

    def forward(self, x):
        out = self.conv1(x)
        return out


class FusionConv(nn.Module):
    def __init__(self, num_features_in, num_features_out):
        super(FusionConv, self).__init__()
        self.conv1 = DoubleConv(in_channels = num_features_in, out_channels = num_features_out)
        self.conv2 = DoubleConv(in_channels = num_features_out, out_channels = num_features_out)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class Conv1_1(nn.Module):
    def __init__(self, num_features_in, num_features_out):
        super(Conv1_1, self).__init__()
        self.conv1 = DoubleConv(num_features_in, num_features_out)

    def forward(self, x):
        out = self.conv1(x)

        return out

class Conv3_3(nn.Module):
    def __init__(self, num_features_in, num_features_out, stride):
        super(Conv3_3, self).__init__()
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_out, kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv3_3(x)
        out = self.relu(out)

        return out

class Conv_Delta(nn.Module):
    def __init__(self, num_anchor_points, feature_size):
        super(Conv_Delta, self).__init__()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, x):
        out = self.output(x)
        return out

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(ClassificationModel, self).__init__()

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out

class Conv_logits(nn.Module):
    def __init__(self, num_anchor_points, signal_classes, feature_size):
        super(Conv_logits, self).__init__()

        self.signal_classes = signal_classes
        self.num_anchor_points = num_anchor_points

        self.output = nn.Conv2d(feature_size, num_anchor_points * signal_classes, kernel_size=(3, 3), stride=2, padding=1)


    def forward(self, x):
        out = self.output(x)

        return out


class HungarianMatcher_Crowd():

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def matcher(self, pred_coords, pred_logits, tgt_points, tgt_ids):
        # print('########### pred_coords size ##########', pred_logits.size())
        roi_num, num_queries = pred_logits.shape[:2]
        out_probs = pred_logits.softmax(-1)  # [roi_num * num_queries, num_classes]
        out_points = pred_coords  # [roi_num * num_queries, 2]
        sizes = [len(label) for label in tgt_ids]
        assert len(sizes) == roi_num

        indices_results = []

        for i, (tgt_point, tgt_label) in enumerate(zip(tgt_points, tgt_ids)):
            out_prob = out_probs[i].cpu()
            out_point = out_points[i].cpu()
            tgt_label = tgt_label.type(torch.long).cpu()
            cost_class = -out_prob[:, tgt_label]
            # print('tgt_point size', tgt_point.size())
            try:
                cost_point = torch.cdist(out_point.float(), tgt_point.cpu().float(), p=2)
                # print('cost_class', cost_class.size())
                # print('cost_point', cost_point.size())
                C = self.cost_point * cost_point + self.cost_class * cost_class
                indices = torch.as_tensor(linear_sum_assignment(C), dtype=torch.int64).long()
                indices_results.append(indices)
            except:
                indices_results.append(torch.as_tensor([]))

        return indices_results


@HEADS.register_module()
class HTCSignalUNetHead(FCNMaskHead):
    def __init__(self, return_logit, num_convs, roi_feat_size, in_channels, conv_kernel_size, conv_out_channels,
                 num_classes, signal_classes, loss_points_coef, loss_cls_coef, signal_loss_weight, eos_coef, red_coef, *args, **kwargs):
        super(HTCSignalUNetHead, self).__init__(num_convs, roi_feat_size, in_channels, conv_kernel_size, conv_out_channels, num_classes, *args, **kwargs)

        self.mask_feature_conv = MaskConv(num_features_in=conv_out_channels, feature_size=conv_out_channels)
        self.conv1_1_0 = Conv1_1(num_features_in=768, num_features_out=384)
        self.fusion_conv_0 = FusionConv(num_features_in=384, num_features_out=384)

        self.conv1_1_1 = Conv1_1(num_features_in=640, num_features_out=320)
        self.fusion_conv_1 = FusionConv(num_features_in=320, num_features_out=320)

        self.Conv3_3 = Conv3_3(num_features_in=256, num_features_out=256, stride=2)

        self.conv1_1_list = [self.conv1_1_0, self.conv1_1_1]
        self.fusion_conv_list = [self.fusion_conv_0, self.fusion_conv_1]

        self.num_anchor_points = 1
        self.eos_coef = eos_coef
        self.red_coef = red_coef
        self.signal_classes = signal_classes + 1
        empty_weight = torch.ones(self.signal_classes)
        empty_weight[0] = self.eos_coef
        empty_weight[1] = self.red_coef
        self.register_buffer('empty_weight', empty_weight)
        self.signal_loss_weight = signal_loss_weight
        self.loss_signal_points_dict = loss_points_coef
        self.loss_signal_labels_dict = loss_cls_coef

        self.loss_signal_points = build_loss(self.loss_signal_points_dict)
        self.loss_signal_labels = build_loss(self.loss_signal_labels_dict)
        self.matcher = HungarianMatcher_Crowd(cost_class=1, cost_point=0.1)

        self.regression_branch = RegressionModel(num_features_in=conv_out_channels, feature_size=conv_out_channels)
        self.conv_delta = Conv_Delta(num_anchor_points=self.num_anchor_points, feature_size=conv_out_channels)

        self.classification_branch = ClassificationModel(num_features_in=conv_out_channels, feature_size=conv_out_channels)
        self.conv_logits = Conv_logits(num_anchor_points=self.num_anchor_points, signal_classes=self.signal_classes,
                                       feature_size=conv_out_channels)


    def forward(self, x, reture_conv3_3 = False):

        for conv in self.convs:
            x = conv(x)

        if reture_conv3_3:
            res_feat = self.Conv3_3(x)
        else:
            res_feat = x

        return res_feat


    def forward_signal(self, x):

        x_delta_feat = self.regression_branch(x)
        deltas = self.conv_delta(x_delta_feat)
        deltas = deltas.permute(0, 2, 3, 1).contiguous()
        deltas = deltas.view(deltas.shape[0], -1, 2)

        x_logits_feat = self.classification_branch(x)
        pred_logits = self.conv_logits(x_logits_feat)
        pred_logits = pred_logits.permute(0, 2, 3, 1).contiguous()
        pred_logits = pred_logits.view(pred_logits.shape[0], -1, self.signal_classes)

        return deltas, pred_logits

    def forward_fusion(self, x, signal_roi_extractor, pos_rois, mask_feature, last_signal_feat = None):

        # print('############### stage start #################')

        if last_signal_feat is not None:
            feature_in = last_signal_feat + mask_feature
            feature_out = self.mask_feature_conv(feature_in)
        else:
            feature_out = mask_feature

        for each_level in range(len(x)):
            cur_level_roi_extractor = signal_roi_extractor[each_level]
            cur_level_feats = x[each_level].unsqueeze(0)
            # print('cur_level_feats size', cur_level_feats.size())
            cur_level_roi_feats = cur_level_roi_extractor(cur_level_feats, pos_rois)
            if each_level == 0:
                upsample_feats = F.interpolate(cur_level_roi_feats, scale_factor=2, mode='bilinear', align_corners=False)
                # print('upsample_feats size', upsample_feats.size())
                last_features_cat = torch.cat([upsample_feats, feature_out], dim=1)
                # print('cur_features_cat size', last_features_cat.size())
            else:
                # print('cur_level_roi_feats, last_features_cat', cur_level_roi_feats.size(), last_features_cat.size())
                cur_features = torch.cat([cur_level_roi_feats, last_features_cat], dim=1)
                # print('cur_features', cur_features.size())
                conv1_1 = self.conv1_1_list[each_level - 1]
                cur_features = conv1_1(cur_features)
                fusion_conv = self.fusion_conv_list[each_level - 1]
                fusion_feature =fusion_conv(cur_features)
                # print('fusion_feature', fusion_feature.size())
                last_features_cat = F.interpolate(fusion_feature, scale_factor=2, mode='bilinear', align_corners=False)
            # print('------------- level end ---------------')

        # print('############### stage end #################')

        return fusion_feature


    def get_targets(self, sampling_results, gt_signal_points, gt_signal_labels):
        """Get training targets of MaskPointHead for all images.
        Returns:
            Tensor: Point target, shape (num_rois, num_points).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        # rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        signal_targets = list(map(self._get_target_single, pos_proposals, pos_assigned_gt_inds_list,
                                  gt_signal_points, gt_signal_labels))

        return signal_targets


    def _get_target_single(self, rois, pos_assigned_gt_inds, gt_signal_points, gt_signal_labels):
        """Get training target of MaskPointHead for each image."""
        num_pos = rois.size(0)
        if num_pos > 0:
            signal_points_within_masks = [gt_signal_points[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]
            signal_labels_within_masks = [gt_signal_labels[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]

            signal_points_within_masks = abs_img_point_to_rel_roi_point(rois, signal_points_within_masks)
            all_signal_points = signal_points_within_masks
            all_signal_labels = signal_labels_within_masks

        else:
            all_signal_points = []
            all_signal_labels = []

        signal_target = {}
        signal_target['points'] = all_signal_points
        signal_target['labels'] = all_signal_labels

        # vis_abs_points(all_target_points=all_signal_points, all_target_labels=all_signal_labels, rois= rois)

        return signal_target

    def _get_src_permutation_idx(self, indices):

        roi_idx_list = []
        src_idx_list = []

        for i, index in enumerate(indices):
            if len(index) > 0:
                roi_idx_list.append(torch.full_like(index[0], i))
                src_idx_list.append(index[0])

        try:
            roi_idx = torch.cat(roi_idx_list)
            src_idx = torch.cat(src_idx_list)
        except:
            roi_idx = []
            src_idx = []

        return roi_idx, src_idx


    def signal_points(self, pred_coords, tgt_points, indices):

        idx = self._get_src_permutation_idx(indices)
        try:
            src_points = pred_coords[idx]
        except:
            src_points = []

        target_points_list = []
        for target_points, index in zip(tgt_points, indices):
            if len(index) > 0:
                # print('index size', index)
                t_id = index[1]
                target_points_list.append(target_points[t_id])
        try:
            target_points = torch.cat(target_points_list)
        except:
            target_points = []

        return src_points, target_points


    def signal_labels(self, pred_logits, tgt_labels, indices):

        src_logits = pred_logits
        idx = self._get_src_permutation_idx(indices)

        target_labels_list = []
        for target_labels, index in zip(tgt_labels, indices):
            if len(index) > 0:
                t_id = index[1]
                target_labels_list.append(target_labels[t_id])

        try:
            target_classes_o = torch.cat(target_labels_list).long()
            target_classes = torch.full(src_logits.shape[:2], 0,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

        except:
            target_classes = torch.full(src_logits.shape[:2], 0,
                                        dtype=torch.int64, device=src_logits.device)

        return src_logits, target_classes


    def loss(self, pred_coords, pred_logits, tgt_points, tgt_labels, indices, indices_candidates = None):

        src_points, target_points = self.signal_points(pred_coords, tgt_points, indices)
        src_logits, target_classes = self.signal_labels(pred_logits, tgt_labels, indices)

        if len(src_points) == 0:
            loss_signal_p = torch.tensor(0).to(pred_coords.device)
            loss_signal_l = torch.tensor(0).to(pred_coords.device)
            print('failed on one image')
        else:
            # vis_match_points(candidate_points, candidate_target_points)
            loss_signal_p = self.loss_signal_points(src_points.float(), target_points.float())
            loss_signal_l = self.loss_signal_labels(src_logits.transpose(1, 2), target_classes, self.empty_weight.cuda(target_classes.device))
        # print('########### src_points #############', src_points[:10])
        # print('########### target_points ##########', target_points[:10])

        # print('########### src_logits #############', torch.argmax(src_logits, dim = 2)[:3])
        # print('########### target_classes ##########', target_classes[:3])

        loss_signal = (loss_signal_p + loss_signal_l) * self.signal_loss_weight
        # loss_signal = loss_signal_l * self.signal_loss_weight
        print('########### loss_signal_point #############', loss_signal_p)
        print('########### loss_signal_class ##########', loss_signal_l)
        loss = dict()
        loss['loss_signal'] = loss_signal

        return loss
