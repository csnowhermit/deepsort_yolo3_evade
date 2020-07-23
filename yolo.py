# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from common.config import special_types, log

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/trained_weights_final-20200709-all-epoch=50.h5',
        # "model_path": 'model_data/yolo_weights.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/evade_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = self.get_defaults("model_image_size")  # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, {} anchors: {}, and {} classes loaded, details: {}'.format(model_path, len(self.anchors), self.anchors, len(self.class_names), self.class_names))
        log.logger.info('{} model, {} anchors: {}, and {} classes loaded, details: {}'.format(model_path, len(self.anchors), self.anchors, len(self.class_names), self.class_names))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        person_classes = []  # 人，用于做tracker
        person_boxs = []
        person_scores = []

        other_classes = []  # 其他物品，只用来显示
        other_boxs = []
        other_scores = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]

            box = out_boxes[i]  # 原始结果：上左下右
            score = out_scores[i]
            print("原始检出：%s %s %s" % (predicted_class, box, score))
            log.logger.info("原始检出：%s %s %s" % (predicted_class, box, score))

            x = int(box[1])    # 左
            y = int(box[0])    # 上
            w = int(box[3] - box[1])    # 右-左：宽
            h = int(box[2] - box[0])    # 下-上：高
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0

            if predicted_class in special_types:  # 人和其他目标分开处理
                person_classes.append(predicted_class)
                person_boxs.append([x, y, w, h])    # 左上宽高
                person_scores.append(score)
            else:    # 其他类别用原始格式的框：上左下右
                other_classes.append(predicted_class)
                other_boxs.append(box)
                other_scores.append(score)
        return (person_classes, person_boxs, person_scores), (other_classes, other_boxs, other_scores)

    def close_session(self):
        self.sess.close()