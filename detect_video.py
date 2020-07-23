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
from common.config import tracker_type, normal_save_path, evade_save_path, ip, table_name
from common.evadeUtil import evade_vote
from common.dateUtil import formatTimestamp
from common.dbUtil import saveManyDetails2DB, getMaxPersonID

warnings.filterwarnings('ignore')

def main(yolo, input_path, output_path):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    curr_person_id = getMaxPersonID(table_name)
    tracker = Tracker(metric, n_start=curr_person_id)    # 用tracker来维护Tracks，每个track跟踪一个人

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

    image_size = (640, 480)    # 图片大小

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image_size[1] + 0.5).astype('int32'))    # 640*480
    thickness = (image_size[0] + image_size[1]) // 300

    video_capture = cv2.VideoCapture(input_path)

    video_FourCC = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    while True:
        read_t1 = time.time()    # 读取动作开始
        print("===================")
        ret, frame = video_capture.read()  # frame shape (h, w, c) (1080, 1920, 3)
        if ret != True:
             break
        read_time = time.time() - read_t1    # 读取动作结束
        detect_t1 = time.time()    # 检测动作开始

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        (person_classes, person_boxs, person_scores), \
        (other_classes, other_boxs, other_scores) = yolo.detect_image(image)    # person_boxs格式：左上宽高
        # print("person:", person_boxs, person_scores)    # 返回的结果均为[x, y, w, h]
        # print("other:", other_classes, other_boxs, other_scores)

        features = encoder(frame, person_boxs)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(person_boxs, person_scores, features)]
        # print("detections:", detections, [det.confidence for det in detections])    # [<deep_sort.detection.Detection object at 0x000001AA02280A90>] [0.9554405808448792]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])    # d.tlwh的格式：左上宽高
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


        # 判定通行状态：0正常通过，1涉嫌逃票
        flag, TrackContentList = evade_vote(tracker.tracks, other_classes, other_boxs, other_scores, frame.shape[0])    # frame.shape, (h, w, c)

        detect_time = time.time() - detect_t1    # 检测动作结束

        # 标注
        draw = ImageDraw.Draw(image)

        for track in tracker.tracks:    # 标注人，track.state=0/1，都在tracker.tracks中
            bbox = track.to_tlbr()    # 左上右下
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
                    outline=colors[class_names.index(tracker_type)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[class_names.index(tracker_type)])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        for (other_cls, other_box, other_score) in zip(other_classes, other_boxs, other_scores):    # 其他的识别，只标注类别和得分值
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

        frame = np.asarray(image)    # 这时转成np.ndarray后是rgb模式
        # bgr = rgb[..., ::-1]    # rgb转bgr
        frame = frame[..., ::-1]
        # cv2.imshow('', frame)
        # cv2.waitKey(1)
        print(time.time() - read_t1)

        ################ 批量入库 ################
        if len(TrackContentList) > 0:    # 只有有人，才进行入库，保存等操作
            curr_time = formatTimestamp(int(read_t1))    # 当前时间按读取时间算
            if flag == "NORMAL":    # 正常情况
                savefile = os.path.join(normal_save_path, ip + "_" + curr_time + ".jpg")
                print(cv2.imwrite(savefile, frame))  # 保存到文件
                # cv2.imencode('.png', frame)[1].tofile(savefile)
            elif flag == "WARNING":    # 逃票情况
                savefile = os.path.join(evade_save_path, ip + "_" + curr_time + ".jpg")
                print(cv2.imwrite(savefile, frame))    # 保存到文件
            else:    # 没人的情况
                pass
            saveManyDetails2DB(ip=ip,
                               curr_time=curr_time,
                               savefile=savefile,
                               read_time=read_time,
                               detect_time=detect_time,
                               predicted_class=tracker_type,
                               TrackContentList=TrackContentList)    # 批量入库
        if isOutput:    # 识别后的视频保存
            out.write(frame)
    yolo.close_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--input", nargs='?', type=str, required=True,
        help="Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if "input" in FLAGS:
        main(YOLO(), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")


