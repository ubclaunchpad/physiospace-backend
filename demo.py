import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")
sys.path.insert(0, "../eval/")

import os
import numpy as np
import scipy.misc
# from heatmap_process import post_process_heatmap
import argparse
# from pckh import run_pckh
from mpii_datagen import MPIIDataGen
import cv2

import net

MODEL_DIR = "models/hg_s2_b1/"

def render_joints(cvmat, joints, conf_threshold=0.2):
    for joint in joints:
        x, y, confidence = joint
        if conf > conf_threshold:
            cv2.circle(cvmat, center=(int(x), int(y)), color=(255, 0, 0), radius=7, thickness=2)

    return cvmat

def infer(model_json, model_weights, num_stack, imgfile, conf_threshold):

    # numClasses, numStacks, numChannels, inputRes, outputRes, learningRate
    hgnet = net.HGNet(numClasses=16, numStacks=num_stack, numChannels=256, inputRes=(256, 256),
                            outputRes=(64, 64), learningRate = 5e-4)

    hgnet.load_model(model_json, model_weights)

    out, scale = hgnet.inference_file(imgfile)

    # kps = post_process_heatmap(out[0, :, :, :])

    # ignore_kps = ['plevis', 'thorax', 'head_top']
    # kp_keys = MPIIDataGen.get_kp_keys()
    # mkps = list()
    # for i, _kp in enumerate(kps):
    #     if kp_keys[i] in ignore_kps:
    #         _conf = 0.0
    #     else:
    #         _conf = _kp[2]
    #     mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

    # cvmat = render_joints(cv2.imread(imgfile), mkps, conf_threshold)

    # cv2.imshow('frame', cvmat)
    # cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", help='input image file')
    parser.add_argument("--conf_threshold", type=float, default=0.2, help='confidence threshold')

    args = parser.parse_args()

    infer(model_json=MODEL_DIR + "net_arch.json", model_weights=MODEL_DIR + "weights_epoch96.h5", num_stack=2,
        imgfile=args.input_image, conf_threshold=args.conf_threshold)