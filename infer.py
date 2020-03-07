import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")
sys.path.insert(0, "../eval/")

import os
import numpy as np
import scipy.misc
from data_process import post_process_heatmap, render_joints
import argparse
from mpii_datagen import MPIIDataGen
import cv2

import net

MODEL_DIR = "models/hg_s2_b1/"


class InferenceEngine():

    def __init__(self, model_json, model_weights):
        # numClasses, numStacks, numChannels, inputRes, outputRes, learningRate
        self.hgnet = net.HGNet(numClasses=16, numStacks=2, numChannels=256, inputRes=(256, 256),
                               outputRes=(64, 64), learningRate=5e-4)

        self.hgnet.load_model(model_json, model_weights)

    def infer(self, imgfile, conf_threshold):
        out, scale = self.hgnet.inference_file(imgfile)

        kps = post_process_heatmap(out[0, :, :, :])

        ignore_kps = ['plevis', 'thorax', 'head_top']
        kp_keys = MPIIDataGen.get_kp_keys()
        mkps = list()
        for i, _kp in enumerate(kps):
            if kp_keys[i] in ignore_kps:
                _conf = 0.0
            else:
                _conf = _kp[2]
            mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

        cvmat = render_joints(cv2.imread(imgfile), mkps, conf_threshold)

        cv2.imshow('frame', cvmat)
        cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", help='input image file')
    parser.add_argument("--conf_threshold", type=float,
                        default=0.2, help='confidence threshold')

    args = parser.parse_args()

    inference = InferenceEngine(
        model_json=MODEL_DIR + "net_arch.json", model_weights=MODEL_DIR + "weights_epoch96.h5")
    inference.infer(imgfile=args.input_image,
                    conf_threshold=args.conf_threshold)
