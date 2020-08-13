#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import time
import warnings
import cv2
import argparse
import traceback
import numpy as np
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image, ImageDraw, ImageFont
import colorsys
from common.config import normal_save_path, evade_save_path, ip, log, image_size, evade_origin_save_path
from common.evadeUtil import evade_vote
from common.dateUtil import formatTimestamp
from common.dbUtil import saveManyDetails2DB, getMaxPersonID
from common.trackUtil import getUsefulTrack

warnings.filterwarnings('ignore')


def main(yolo, input_path):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    curr_person_id = getMaxPersonID()
    tracker = Tracker(metric, n_start=curr_person_id)  # 用tracker来维护Tracks，每个track跟踪一个人

    class_names = yolo._get_class()

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image_size[1] + 0.5).astype('int32'))  # 640*480
    thickness = (image_size[0] + image_size[1]) // 300

    frame = cv2.imread(input_img)
    # ret, frame = cap.read()  # frame shape (h, w, c) (1080, 1920, 3)

    # image = Image.fromarray(frame)
    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    # 大人和小孩分开处理，这时已进行nms
    (adult_classes, adult_boxs, adult_scores), \
    (child_classes, child_boxs, child_scores), \
    (other_classes, other_boxs, other_scores) = yolo.detect_image(image)  # person_boxs格式：左上宽高
    # print("person/child:", person_boxs, person_scores)    # 返回的结果均为[x, y, w, h]
    # print("other:", other_classes, other_boxs, other_scores)

    # 每个人做成128维的向量
    features_adult = encoder(frame, adult_boxs)  # 大人
    features_child = encoder(frame, child_boxs)  # 小孩
    # print("features:", type(features), features.shape)    # <class 'numpy.ndarray'> (n, 128)

    # 大人和小孩的Detection
    detections_adult = [Detection(cls, bbox, confidence, feature) for cls, bbox, confidence, feature in
                        zip(adult_classes, adult_boxs, adult_scores, features_adult)]
    detections_child = [Detection(cls, bbox, confidence, feature) for cls, bbox, confidence, feature in
                        zip(child_classes, child_boxs, child_scores, features_child)]

    # # 原来有nms，现去掉，原因：yolo.detect_image()本身已经做了nms
    # boxes = np.array([d.tlwh for d in detections])
    # scores = np.array([d.confidence for d in detections])
    # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    # detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections_adult + detections_child)  # Detection中有区分大人小孩了，这里直接放一起追踪

    # 这里出现bug：误检，只检出一个人，为什么tracker.tracks中有三个人
    # 原因：人走了，框还在
    # 解决办法：更新后的tracker.tracks与person_boxs再做一次iou，对于每个person_boxs，只保留与其最大iou的track

    trackList_adult = getUsefulTrack(adult_boxs, tracker.tracks)
    trackList_child = getUsefulTrack(child_boxs, tracker.tracks)

    print("检测到：大人 %d %s, 小孩 %d %s" % (len(adult_boxs), adult_boxs, len(child_boxs), child_boxs))
    print("追踪到：大人 %d %s, 小孩 %d %s" % (len(trackList_adult), [track.to_tlbr() for track in trackList_adult],
                                      len(trackList_child), [track.to_tlbr() for track in trackList_child]))
    log.logger.info("检测到：大人 %d %s, 小孩 %d %s" % (len(adult_boxs), adult_boxs, len(child_boxs), child_boxs))
    log.logger.info(
        "追踪到：大人 %d %s, 小孩 %d %s" % (len(trackList_adult), [track.to_tlbr() for track in trackList_adult],
                                    len(trackList_child), [track.to_tlbr() for track in trackList_child]))

    trackList = trackList_adult + trackList_child
    # 判定通行状态：0正常通过，1涉嫌逃票
    flag, TrackContentList = evade_vote(trackList, other_classes, other_boxs, other_scores,
                                        frame.shape[0])  # frame.shape, (h, w, c)


    # 标注
    draw = ImageDraw.Draw(image)

    for track in trackList:  # 标注人，track.state=0/1，都在tracker.tracks中
        bbox = track.to_tlbr()  # 左上右下
        label = '{} {:.2f} {} {}'.format(track.classes, track.score, track.track_id, track.state)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = bbox
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        log.logger.info("%s, (%d, %d), (%d, %d)" % (label, left, top, right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[class_names.index(track.classes)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[class_names.index(track.classes)])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    for (other_cls, other_box, other_score) in zip(other_classes, other_boxs, other_scores):  # 其他的识别，只标注类别和得分值
        label = '{} {:.2f}'.format(other_cls, other_score)
        # print("label:", label)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = other_box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        log.logger.info("%s, (%d, %d), (%d, %d)" % (label, left, top, right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[class_names.index(other_cls)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[class_names.index(other_cls)])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw

    result = np.asarray(image)  # 这时转成np.ndarray后是rgb模式，out.write(result)保存为视频用
    # bgr = rgb[..., ::-1]    # rgb转bgr
    result = result[..., ::-1]

    cv2.imshow("detect_img", result)
    cv2.waitKey()

    yolo.close_session()


if __name__ == '__main__':
    input_img = "C:/Users/vegetabledog/Desktop/10.6.8.181_20200812_093905.445-origin.jpg"
    main(YOLO(), input_img)


