import torch
from copy import deepcopy
from shapely import geometry
import poisson_disc as pd
import numpy as np
import cv2
import random
from imageio import imsave


def abs_img_point_to_rel_roi_point(rois_th, signal_points_within_mask):
    """Convert image based absolute point coordinates to roi based relative point coordinates.
    """
    with torch.no_grad():
        abs_img_points = deepcopy(signal_points_within_mask)
        signal_points_rel_roi_points = []
        for each_roi_points, each_roi in zip(abs_img_points, rois_th):
            if len(each_roi_points) > 0:
                if isinstance(each_roi_points, (tuple, list)):
                    each_roi_points = each_roi_points[0]
                xs = each_roi_points[:, 0] - each_roi[0]
                ys = each_roi_points[:, 1] - each_roi[1]
                coords = torch.stack([xs, ys], dim=1)
                signal_points_rel_roi_points.append(coords)
            else:
                signal_points_rel_roi_points.append(torch.tensor([]).cuda(rois_th.device))

    return signal_points_rel_roi_points


def abs_ignore_to_rel_roi_point(rois, signal_ignore_area_within_masks):
    with torch.no_grad():
        abs_ignore_boundary = deepcopy(signal_ignore_area_within_masks)
        roi_ignores = []
        for each_roi_ignores, each_roi in zip(abs_ignore_boundary, rois):
            if len(each_roi_ignores) > 0:
                ignores_list = []
                for each_roi_ignore in each_roi_ignores:
                    # print('each_roi_ignore', each_roi_ignore)
                    xs = each_roi_ignore[:, 0] - each_roi[0]
                    ys = each_roi_ignore[:, 1] - each_roi[1]
                    coords = torch.stack([xs, ys], dim=1)
                    ignores_list.append(coords)
                roi_ignores.append(ignores_list)
            else:
                roi_ignores.append(torch.tensor([]).cuda(each_roi.device))

    return roi_ignores


def dense_sampling(rois, signal_ignore_rel_roi_area):
    sampling_points = []
    stride = 4

    for roi_ignores, each_roi in zip(signal_ignore_rel_roi_area, rois):

        if len(roi_ignores) > 0:
            roi_ignores = [ignore_area.unsqueeze(0) for ignore_area in roi_ignores]
            poly_context = {'type': 'MULTIPOLYGON', 'coordinates': roi_ignores}
            poly = geometry.shape(poly_context)

            dims2d = np.array(
                [int(each_roi[2].item() - each_roi[0].item()), int(each_roi[3].item()) - int(each_roi[1].item())])
            points_data = pd.Bridson_sampling(dims=dims2d, radius=stride, k=30,
                                              hypersphere_sample=pd.hypersphere_surface_sample)
            points_data = torch.tensor(points_data).cuda(each_roi.device)
            # print('############## points_data ############', points_data.size())
            try:
                coords = torch.stack([torch.stack([x, y]) for x, y in zip(points_data[:, 0], points_data[:, 1]) if
                                      geometry.Point(x, y).within(poly)])
                sampling_points.append(coords)
            except:
                sampling_points.append(torch.tensor([]).cuda(each_roi.device))

        else:
            sampling_points.append(torch.tensor([]).cuda(each_roi.device))

    return sampling_points

def vis_points(all_pred_points, all_pred_logits, rois):

    # print('all_pred_points size', all_pred_points.size())
    # print('all_pred_logits size', all_pred_logits.size())
    rois = rois[:, 1:]
    print('rois size', rois.size())

    all_pred_labels = torch.argmax(all_pred_logits, dim=-1)
    assert len(all_pred_points) == len(all_pred_labels)
    vis_points_image = np.uint8(np.zeros((800, 1280, 3)))
    randfloat = random.randint(1, 100)
    label_color_dict = {'0':(255, 255, 0), '1': (255, 0, 0), '2': (0, 255, 0)}
    for pred_rel_points, pred_labels, roi in zip(all_pred_points, all_pred_labels, rois):
        for pred_rel_point, pred_label in zip(pred_rel_points, pred_labels):
            # if int(pred_label) > 0:
            vis_points_image = cv2.circle(vis_points_image, (
            int(pred_rel_point[0] + int(roi[0])), int(pred_rel_point[1]) + int(roi[1])),
                                          radius=1, color=label_color_dict[str(int(pred_label))],
                                          thickness=-1)
    for roi in rois:
        vis_points_image = cv2.rectangle(vis_points_image, (int(roi[0]), int(roi[1])),
                                         (int(roi[2]), int(roi[3])),
                                         (255, 255, 0), 2)
    print('randfloat', randfloat)
    imsave('/data2/Caijt/FISH_mmdet/check_sanity/pred_vis/' + str(randfloat) + '.jpg', vis_points_image)


def vis_match_points(candidate_points, candidate_target_points):
    vis_points_image = np.uint8(np.zeros((800, 1280, 3)))
    randfloat = random.randint(1, 100)
    label_color_dict = {'matched':(255, 0, 0), 'candidates': (0, 255, 0)}
    for candidate_target_point in candidate_target_points:
        vis_points_image = cv2.circle(vis_points_image, candidate_target_point,
                                      radius=1, color=label_color_dict['matched'],
                                      thickness=-1)
    for candidate_point in candidate_points:
        vis_points_image = cv2.circle(vis_points_image, candidate_point,
                                      radius=1, color=label_color_dict['candidates'],
                                      thickness=-1)

    print('randfloat', randfloat)
    imsave('/data2/Caijt/FISH_mmdet/check_sanity/match_vis/' + str(randfloat) + '.jpg', vis_points_image)



def vis_abs_points(all_target_points, all_target_labels, rois):
    # print('==================== len =================', len(rois), len(center_heatmap_target), len(ignore_region_target))
    assert len(all_target_points) == len(all_target_labels)
    vis_points_image = np.uint8(np.zeros((800, 1280, 3)))
    original_image = vis_points_image
    randfloat = random.randint(1, 100)
    label_color_dict = {'1': (255, 0, 0), '2': (0, 255, 0)}
    for target_rel_points, traget_labels, roi in zip(all_target_points, all_target_labels, rois):
        for target_rel_point, target_label in zip(target_rel_points, traget_labels):
            if int(target_label) > 0:
                vis_points_image = cv2.circle(vis_points_image, (int(target_rel_point[0] + int(roi[0])), int(target_rel_point[1])+int(roi[1])),
                                              radius = 1, color = label_color_dict[str(int(target_label))], thickness = -1)
    for roi in rois:
        vis_points_image = cv2.rectangle(vis_points_image, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])),
                                         (255, 255, 0), 2)
    print('randfloat', randfloat)
    imsave('/data2/Caijt/FISH_mmdet/check_sanity/target_vis/' + str(randfloat) + '.jpg', vis_points_image)
    # # imsave('/data2/Caijt/FISH_mmdet/check_sanity/target_vis/' + str(randfloat) + '_original.jpg', original_image)
    #
