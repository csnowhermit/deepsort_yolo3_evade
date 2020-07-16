#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image, ImageDraw, ImageFont
import colorsys
from common.Stack import Stack
from common.config import rtsp_url
import threading

warnings.filterwarnings('ignore')

'''
    视频流读取线程：读取到自定义缓冲区
'''
def capture_thread(input_webcam, frame_buffer, lock):
    if input_webcam == "0":
        input_webcam = int(0)
    print("capture_thread start")
    vid = cv2.VideoCapture(input_webcam)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        if return_value is not True:
            break
        lock.acquire()
        frame_buffer.push(frame)
        lock.release()
        cv2.waitKey(25)    # delay 25ms


def detect_thread(frame_buffer, lock):
    yolo = YOLO()

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)  # 用tracker来维护Tracks，每个track跟踪一个人

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

    while True:
        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()    # 每次拿最新的
            lock.release()

            t1 = time.time()

            # image = Image.fromarray(frame)
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            (person_classes, person_boxs, person_scores), \
            (other_classes, other_boxs, other_scores) = yolo.detect_image(image)
            # print("person:", person_boxs, person_scores)    # 返回的结果均为[x, y, w, h]
            # print("other:", other_classes, other_boxs, other_scores)

            features = encoder(frame, person_boxs)

            detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                          zip(person_boxs, person_scores, features)]
            # print("detections:", detections, [det.confidence for det in detections])    # [<deep_sort.detection.Detection object at 0x000001AA02280A90>] [0.9554405808448792]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            # print("====1.", boxes)
            scores = np.array([d.confidence for d in detections])
            # print("====2.", scores)
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            # print("====3.", indices)
            detections = [detections[i] for i in indices]
            # for det in detections:
            #     print("====4.", det.confidence, det.to_tlbr(), end=" ")
            # print("\n")

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # 标注
            draw = ImageDraw.Draw(image)

            for track in tracker.tracks:  # 标注人，track.state=0/1，都在tracker.tracks中
                bbox = track.to_tlbr()  # 左上右下
                # print("==循环中。。", bbox)
                label = '{} {:.2f} {} {}'.format("head", track.score, track.track_id, track.state)
                # print("++++++++++++++++++++++++++++++++++++label:", label)
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
                        outline=colors[class_names.index("person")])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[class_names.index("person")])
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

            frame = np.asarray(image)  # 这时转成np.ndarray后是rgb模式
            # bgr = rgb[..., ::-1]    # rgb转bgr
            frame = frame[..., ::-1]
            cv2.imshow('', frame)
            cv2.waitKey(1)
            print(time.time() - t1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_buffer = Stack(30 * 5)
    lock = threading.RLock()
    t1 = threading.Thread(target=capture_thread, args=(rtsp_url, frame_buffer, lock))
    t1.start()
    t2 = threading.Thread(target=detect_thread, args=(frame_buffer, lock))
    t2.start()
