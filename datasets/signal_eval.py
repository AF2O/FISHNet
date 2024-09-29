import numpy as np
import time
import copy
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import random
import cv2
from imageio import imsave

class Params:
    '''
    Params for coco evaluation api
    '''
    def __init__(self, iouType='signal'):
        self.imgIds = []
        self.catIds = []
        self.iouType = iouType

class SIGNALeval:

    def __init__(self, cocoGt=None, cocoDt=None, iouType='signal'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None

        '''
        if not iouType:
            print('iouType not specified. use default iouType signal')

        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.label_dict = {'1': '红色信号点', '2': '绿色信号点'}

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

        # convert ground truth to mask if iouType == 'segm'
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval = {}                  # accumulated evaluation results


    def get_match_coords(self, pred_center_coords, pred_labels, anno_center_coords, anno_labels, num_classes):


        cost_point = cdist(pred_center_coords, anno_center_coords)
        indices = linear_sum_assignment(cost_point)
        pred_idx, target_id = indices

        dist_threshold = 8
        dist = np.linalg.norm(pred_center_coords[pred_idx] - anno_center_coords[target_id], ord=2, axis=1)
        dist_bool = (dist < dist_threshold)

        ## get pred
        all_pred_num_per_class_list = []
        for i in range(1, num_classes+ 1):
            num_per_class = (pred_labels == i).sum()
            all_pred_num_per_class_list.append(num_per_class)
            # print('all_pred_num_{}: '.format(self.label_dict[str(i)]), num_per_class)

        ## get target
        all_target_num_per_class_list = []
        for i in range(1, num_classes+1):
            num_per_class = (anno_labels == i).sum()
            all_target_num_per_class_list.append(num_per_class)
            # print('target_num_{}: '.format(self.label_dict[str(i)]), num_per_class)

        pred_match_points = pred_center_coords[pred_idx]
        pred_match_labels = pred_labels[pred_idx]
        target_match_labels = anno_labels[target_id]
        label_match_bool = (pred_match_labels == target_match_labels)

        match_bool = label_match_bool * dist_bool
        match_bool = np.array(match_bool)

        if match_bool.sum() > 0:
            pred_match_points = pred_match_points[match_bool]
            pred_match_labels = pred_match_labels[match_bool]
            match_cls_num_per_class_list = []
            for i in range(1, num_classes+1):
                num_match_per_class = (pred_match_labels == i).sum()
                match_cls_num_per_class_list.append(num_match_per_class)
                # print('match_cls_num_{}: '.format(self.label_dict[str(i)]), num_match_per_class)
        else:
            pred_match_points = None
            pred_match_labels = None
            match_cls_num_per_class_list = [0, 0]
            # print('match_cls_num_all: 0')

        stat_dict = {'pred_num': np.array(all_pred_num_per_class_list),
                     'target_num': np.array(all_target_num_per_class_list),
                     'match_cls_num': np.array(match_cls_num_per_class_list)}

        return pred_match_points, pred_match_labels, stat_dict

    def match_results_vis(self, pred_match_points, pred_match_labels, anno_center_coords, anno_labels):

        assert len(pred_match_points) == len(pred_match_labels)
        assert len(anno_center_coords) == len(anno_labels)
        vis_points_image = np.uint8(np.zeros((1200, 1920, 3)))
        randfloat = random.randint(1, 100)
        label_color_dict = {'1': (255, 0, 0), '2': (0, 255, 0)}
        for pred_point, pred_label in zip(pred_match_points, pred_match_labels):
            if int(pred_label) > 0:
                vis_points_image = cv2.circle(vis_points_image, (int(pred_point[0]), int(pred_point[1])),
                                              radius=2, color=label_color_dict[str(int(pred_label))],
                                              thickness=3)
        for anno_point, anno_label in zip(anno_center_coords, anno_labels):
            vis_points_image = cv2.circle(vis_points_image, (int(anno_point[0]), int(anno_point[1])),
                                          radius=5, color=label_color_dict[str(int(anno_label))],
                                          thickness=1)

        imsave('/data2/Caijt/FISH_mmdet/check_sanity/eval_check_match/' + str(randfloat) + '.jpg', vis_points_image)

    def evaluateImg(self, imgId, catId):

        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) ==0:
            return None

        anno_center_coords = [g['signal_points'] for g in gt]
        anno_labels = [g['signal_labels'] for g in gt]

        anno_center_coords = np.concatenate(
            [each_bbox_signal_coords for each_bbox_signal_coords in anno_center_coords if
             len(each_bbox_signal_coords) > 0], axis=0)
        anno_labels = np.concatenate(
            [each_bbox_signal_labels for each_bbox_signal_labels in anno_labels if
             len(each_bbox_signal_labels) > 0], axis=0)

        pred_center_coords = np.concatenate([each_pred_coord['signal_points'] for each_pred_coord in dt], axis = 0)
        pred_labels = np.concatenate([each_pred_coord['signal_labels'] for each_pred_coord in dt], axis = 0)

        pred_match_points, pred_match_labels, stat_dict = \
            self.get_match_coords(pred_center_coords, pred_labels,
                                  anno_center_coords, anno_labels, num_classes = 2)

        # self.match_results_vis(pred_match_points, pred_match_labels, anno_center_coords, anno_labels)

        return {
            'image_id': imgId,
            'category_id': catId,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'pred_match_points': pred_match_points,
            'pred_match_labels': pred_match_labels,
            'stat_dict': stat_dict
        }


    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        # add backward compatibility if useSegm is specified in params
        p = self.params
        print('Evaluate annotation type *{}*'.format(p.iouType))


        self._prepare()
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        print('imgIds', p.imgIds, '\n')
        print('catIds', p.catIds, '\n')
        # loop through images, area range, max detection number

        self.evalImgs = [self.evaluateImg(imgId, catId)
                         for catId in p.catIds
                         for imgId in p.imgIds]

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))



    def accumulate(self, p = None, eps = 1e-5):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params

        # create dictionary for future indexing
        _pe = self._paramsEval
        setI = set(_pe.imgIds)
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        E = [self.evalImgs[i] for i in i_list]
        E = [e for e in E if not e is None]

        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        tp_red_sum, pred_red_sum, target_red_sum = [
            sum(np.concatenate([[e['stat_dict'][x][0]] for e in E]))
            for x in ['match_cls_num', 'pred_num', 'target_num']]


        tp_green_sum, pred_green_sum, target_green_sum = [
            sum(np.concatenate([[e['stat_dict'][x][1]] for e in E]))
            for x in ['match_cls_num', 'pred_num', 'target_num']]

        red_precision, red_recall = tp_red_sum / (pred_red_sum + eps), tp_red_sum / (target_red_sum + eps)

        green_precision, green_recall = tp_green_sum / (pred_green_sum + eps), tp_green_sum / (target_green_sum + eps)

        self.eval = {'red_precision': red_precision,
                     'red_recall': red_recall,
                     'red_F1_score':  2 * red_precision * red_recall / (red_precision + red_recall + eps),
                     'green_precision': green_precision,
                     'green_recall': green_recall,
                     'green_F1_score':  2 * green_precision * green_recall / (green_precision + green_recall + eps)}

        print('eval_results', self.eval)

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

