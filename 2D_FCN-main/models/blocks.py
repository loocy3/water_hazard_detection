import tensorflow_addons as tfa
import tensorflow as tf

from tensorflow.keras.layers import *

def conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
    x = Conv3D(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
    x = tfa.layers.GroupNormalization(groups=32, axis=4)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)
    return x


def ResBlock(U, inp):
    shortcut = inp
    shortcut = conv3d_bn(shortcut, U, 1, 1, 1, activation=None, padding='same')
    conv3x3 = conv3d_bn(inp, U, 3, 3, 3, activation='relu', padding='same')
    conv5x5 = conv3d_bn(conv3x3, U, 3, 3, 3, activation='relu', padding='same')
    # out = BatchNormalization(axis=4)(conv5x5)
    out = tfa.layers.GroupNormalization(groups=32, axis=4)(conv5x5)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = tfa.layers.GroupNormalization(groups=32, axis=4)(out)

    return out


def CSPRes(U, inp, pool_size):
    part1 = inp
    part1 = conv3d_bn(part1, U // 2, 1, 1, 1, activation='relu', padding='same')
    par1 = tfa.layers.GroupNormalization(groups=32, axis=4)(part1)
    part2 = ResBlock(U // 2, inp)

    out = tf.keras.layers.concatenate([part1, part2], axis=4)
    out = tfa.layers.GroupNormalization(groups=32, axis=4)(out)
    return out


def ResPath(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv3d_bn(shortcut, filters, 1, 1, 1, activation=None, padding='same')

    out = conv3d_bn(inp, filters, 3, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = tfa.layers.GroupNormalization(groups=32, axis=4)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv3d_bn(shortcut, filters, 1, 1, 1, activation=None, padding='same')

        out = conv3d_bn(out, filters, 3, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = tfa.layers.GroupNormalization(groups=32, axis=4)(out)

    return out
