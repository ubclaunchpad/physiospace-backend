from keras.models import *
from keras.layers import *
import keras.backend

# full hourglass module
def hourglassModule(bottom, hourglassLayerID, numChannels, numClasses):
    # create downscale blocks (left side of model)
    leftHalf = downscaleBlocks(bottom, hourglassLayerID, numChannels)

    # create right features, connect with left features
    rightHalf = connectingBlocks(
        leftHalf, hourglassLayerID, numChannels)

    # add 1x1 conv with two heads, nextLayer is sent to next stage
    nextLayer, endBlocks = createEndBlocks(
        bottom, rightHalf, numClasses, hourglassLayerID, numChannels)

    return nextLayer, endBlocks

# single downsample block, if numOutChannels is same shape as input, keep shape but apply Conv2D
def downsample(bottom, numOutChannels, blockName):
    # skip layer if dims are same
    if keras.backend.int_shape(bottom)[-1] == numOutChannels:
        skip = bottom
    else:
        skip = SeparableConv2D(numOutChannels, kernel_size=(1, 1), activation='relu', padding='same',
                               name=blockName + 'skip')(bottom)

    # residual: 3 conv blocks,  [numOutChannels/2  -> numOutChannels/2 ->
    # numOutChannels]
    x = SeparableConv2D(numOutChannels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                        name=blockName + '_conv_1x1x1')(bottom)
    x = BatchNormalization()(x)
    x = SeparableConv2D(numOutChannels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                        name=blockName + '_conv_3x3x2')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(numOutChannels, kernel_size=(1, 1), activation='relu', padding='same',
                        name=blockName + '_conv_1x1x3')(x)
    x = BatchNormalization()(x)
    x = Add(name=blockName + '_residual')([skip, x])

    return x

# main downscaling blocks
def downscaleBlocks(bottom, hourglassLayerID, numChannels):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg' + str(hourglassLayerID)

    # 1x resolution
    lf1 = downsample(bottom, numChannels, hgname + '_lf1')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(lf1)

    # 1/2x resolution
    lf2 = downsample(x, numChannels, hgname + '_lf2')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(lf2)

    # 1/4x resolution
    lf4 = downsample(x, numChannels, hgname + '_lf4')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(lf4)

    # 1/8x resolution
    lf8 = downsample(x, numChannels, hgname + '_lf8')

    return (lf1, lf2, lf4, lf8)

# create residual connection blocks
def connectingBlocks(leftLayers, hourglassLayerID, numChannels):
    # unpack left layers
    lf1, lf2, lf4, lf8 = leftLayers

    rf8 = midLayer(lf8, downsample, hourglassLayerID, numChannels)

    rf4 = connection(lf4, rf8, downsample, 'hg' +
                     str(hourglassLayerID) + '_rf4', numChannels)
    rf2 = connection(lf2, rf4, downsample, 'hg' +
                     str(hourglassLayerID) + '_rf2', numChannels)
    rf1 = connection(lf1, rf2, downsample, 'hg' +
                     str(hourglassLayerID) + '_rf1', numChannels)

    return rf1

# end blocks
def createEndBlocks(prelayerfeatures, rf1, num_classes, hgid, num_channels):
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same', name=str(hgid) + '_conv_1x1_x1')(
        rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hgid) + '_conv_1x1_parts')(head)

    # use linear activation
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                  name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hgid) + '_conv_1x1_x3')(head_parts)

    head_next_stage = Add()([head, head_m, prelayerfeatures])
    return head_next_stage, head_parts


# smallest layer in the middle of hourglass
def midLayer(lf8, downsample, hgid, numChannels):
    lf8Connect = downsample(lf8, numChannels, str(hgid) + "_lf10")

    x = downsample(lf8, numChannels, str(hgid) + "_lf10x1")
    x = downsample(x, numChannels, str(hgid) + "_lf10x2")
    x = downsample(x, numChannels, str(hgid) + "_lf10x3")

    rf10 = Add()([x, lf8Connect])

    return rf10


# first 3 layers of model, initial downsampling
def buildInitialDownsample(input, numChannels):
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1x1')(
        input)
    x = BatchNormalization()(x)

    x = downsample(x, numChannels // 2, 'front_residualx1')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = downsample(x, numChannels // 2, 'front_residualx2')
    x = downsample(x, numChannels, 'front_residualx3')

    return x


# create connection between residuals and main
def connection(left, right, downsample, name, numChannels):
    _xleft = downsample(left, numChannels, name + '_connect')
    _xright = UpSampling2D()(right)

    add = Add()([_xleft, _xright])
    out = downsample(add, numChannels, name + '_connect_conv')

    return out
