import yaml
import HG_modules
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error


class HGNet():

    def __init__(self, cfg):

        # read cfg from yaml
        with open(cfg) as file:
            cfgs = yaml.load(file, Loader=yaml.FullLoader)

            self.numClasses = int(cfgs['numClasses'])
            self.numStacks = int(cfgs['hourGlassStackSize'])
            self.numChannels = int(cfgs['numChannels'])
            self.inputRes = (int(cfgs['inputResolutionX']), int(
                cfgs['inputResolutionY']))
            self.outputRes = int(cfgs['outputResolution'])

            self.learningRate = float(cfgs['learningRate'])

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

hg = HGNet("config.yml")
