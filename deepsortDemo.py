#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import time
import warnings
import cv2
import argparse
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image, ImageDraw, ImageFont
import colorsys
from common.config import tracker_type
from common.evadeUtil import calc_iou

warnings.filterwarnings('ignore')

'''
    deepsort demo
'''

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    curr_person_id = 0
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

    image_size = (640, 480)  # 图片大小

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image_size[1] + 0.5).astype('int32'))  # 640*480
    thickness = (image_size[0] + image_size[1]) // 300

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()  # frame shape (h, w, c) (1080, 1920, 3)
        if ret != True:
            break

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        (person_classes, person_boxs, person_scores), \
        (other_classes, other_boxs, other_scores) = yolo.detect_image(image)  # person_boxs格式：左上宽高
        print(tracker_type, person_boxs, person_scores)    # 返回的结果为[x, y, w, h]
        # print("other:", other_classes, other_boxs, other_scores)    # 返回的结果为：上左下右

        features = encoder(frame, person_boxs)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                      zip(person_boxs, person_scores, features)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        trackList = []  # 新的trackList

        if len(person_boxs) > 0:
            person_boxs_ltbr = [[person[0],
                                 person[1],
                                 person[0] + person[2],
                                 person[1] + person[3]] for person in person_boxs]  # person_boxs：左上宽高-->左上右下

            track_box = [[int(track.to_tlbr()[0]),
                          int(track.to_tlbr()[1]),
                          int(track.to_tlbr()[2]),
                          int(track.to_tlbr()[3])] for track in tracker.tracks]  # 追踪器中的人
            if len(track_box) > 0:    # 避免因检测到而未被追踪导致calc_iou出错
                iou_result = calc_iou(bbox1=person_boxs_ltbr, bbox2=track_box)  # 计算iou，做无效track框的过滤
                which_track = []
                for iou in iou_result:
                    if iou.max() > 0.45:  # 如果最大iou>0.45，则认为是这个人
                        which_track.append(iou.argmax())  # 保存tracker.tracks中该人的下标

                # 在tracker.tracks中移除不在which_track的元素
                for i in range(len(tracker.tracks)):
                    if i in which_track:
                        trackList.append(tracker.tracks[i])

        print("检测到 %d 人: %s" % (len(person_boxs), person_boxs))
        print("追踪到 %d 人: %s" % (len(trackList), trackList))

        # 标注
        draw = ImageDraw.Draw(image)

        for track in trackList:  # 标注人，track.state=0/1，都在tracker.tracks中
            bbox = track.to_tlbr()  # 左上右下
            label = '{} {:.2f} {} {}'.format("head", track.score, track.track_id, track.state)
            label_size = draw.textsize(label, font)

            left, top, right, bottom = bbox
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[class_names.index(tracker_type)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[class_names.index(tracker_type)])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

        result = np.asarray(image)  # 这时转成np.ndarray后是rgb模式，out.write(result)保存为视频用
        # bgr = rgb[..., ::-1]    # rgb转bgr
        result = result[..., ::-1]

        cv2.imshow("deepsortDemo", result)
        cv2.waitKey(1)
    yolo.close_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    FLAGS = parser.parse_args()

    main(YOLO())


