import yaml
import HG_modules
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from keras.models import load_model, model_from_json
from matplotlib.pyplot import imread
import numpy as np
from data_process import normalize
from PIL import Image

class HGNet():

    def __init__(self, numClasses, numStacks, numChannels, inputRes, outputRes, learningRate):
        self.numClasses = numClasses
        self.numStacks = numStacks
        self.numChannels = numChannels
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.learningRate = learningRate

        self.model = self.build_model()

        # show model summary
        self.model.summary()

    def build_model(self):
        # define input layer
        inp = Input(shape=(self.inputRes[0], self.inputRes[1], 3))

        # create initial downsample
        frontFeatures = HG_modules.buildInitialDownsample(
            inp, self.numChannels)
        nextLayer = frontFeatures

        # build stack
        net = []
        for i in range(self.numStacks):
                # create hourglass modules based on stack number
            nextLayer, layer = HG_modules.hourglassModule(
                nextLayer, i, self.numChannels, self.numClasses)
            net.append(layer)

        # buld model
        model = Model(inputs=inp, outputs=net)

        # define optimizer
        optimizer = RMSprop(lr=self.learningRate)

        # compile model
        model.compile(optimizer=optimizer,
                      loss=mean_squared_error, metrics=["accuracy"])

        return model

    def load_model(self, model_json, model_file):
        with open(model_json) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(model_file)

    def inference_rgb(self, rgbdata, orgshape, mean=None):
        scale = (orgshape[0] * 1.0 / self.inputRes[0], orgshape[1] * 1.0 / self.inputRes[1])

        imgdata = np.array(Image.fromarray(rgbdata).resize(self.inputRes))

        if mean is None:
            mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]

        out = self.model.predict(input)
        return out[-1], scale

    def inference_file(self, imgfile, mean=None):
        imgdata = imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret


if __name__ == "__main__":
    # read cfg from yaml
    with open("config.yml") as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)

        numClasses = int(cfgs['numClasses'])
        numStacks = int(cfgs['hourGlassStackSize'])
        numChannels = int(cfgs['numChannels'])
        inputRes = (int(cfgs['inputResolutionX']), int(cfgs['inputResolutionY']))
        outputRes = int(cfgs['outputResolution'])

        learningRate = float(cfgs['learningRate'])

        hg = HGNet(numClasses, numStacks, numChannels, inputRes, outputRes, learningRate)