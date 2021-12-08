from models.blocks import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def make_model(input_shape, n_classes, filter_size, kernel_size, pool_size):
    # This code structure is based on Image Segmentation Keras by divamgupta.
    # https://github.com/divamgupta/image-segmentation-keras
    img_input = Input(shape=(input_shape), name='input')
    # Block1
    part1, part2 = tf.split(img_input, 2, 1)
    part1 = ResBlock(filter_size, part1)
    part2 = conv3d_bn(part2, filter_size, 3, 3, 3, padding='same', strides=(1, 1, 1), activation='relu', name=None)
    x = concatenate([part1, part2], axis=1)
    x = tfa.layers.GroupNormalization(groups=32)(x)
    # x = BatchNormalization(name='bn1')(x)
    x = MaxPooling3D(pool_size=pool_size, name='block1_pool1')(x)
    part1, part2 = tf.split(x, 2, 1)
    part1 = ResPath(filter_size, 2, part1)
    part2 = conv3d_bn(part2, filter_size, 3, 3, 3, padding='same', strides=(1, 1, 1), activation='relu', name=None)
    f1 = concatenate([part1, part2], axis=1)
    # Block2
    part1, part2 = tf.split(x, 2, 1)
    part1 = ResBlock(filter_size, part1)
    x = concatenate([part1, part2], axis=1)

    x = tfa.layers.GroupNormalization(groups=32)(x)
    x = MaxPooling3D(pool_size=pool_size, name='block2_pool1')(x)

    part1, part2 = tf.split(x, 2, 1)
    part1 = ResPath(filter_size * 2, 1, part1)
    part2 = conv3d_bn(part2, filter_size * 2, 3, 3, 3, padding='same', strides=(1, 1, 1), activation='relu', name=None)
    f2 = concatenate([part1, part2], axis=1)
    # Block3
    #     x = CSPRes(filter_size*4, x, pool_size)
    #     x = tfa.layers.GroupNormalization(groups=32)(x)
    #     x = MaxPooling3D(pool_size = pool_size, name = 'block3_pool1')(x)
    part1, part2 = tf.split(x, 2, 1)
    part1 = ResBlock(filter_size, part1)
    x = concatenate([part1, part2], axis=1)
    x = tfa.layers.GroupNormalization(groups=32)(x)
    x = MaxPooling3D(pool_size=pool_size, name='block3_pool1')(x)
    f3 = x

    f3_shape = Model(img_input, f3).output_shape
    o = f3
    o = Reshape(f3_shape[2:], name='reshape1')(o)  # remove the first dimension as it equals 1
    # Extract feature
    o = Conv2D(filters=2048, kernel_size=(7, 7), activation='relu', padding='same', data_format='channels_last',
               name='increase_conv1')(o)
    o = Dropout(0.5, name='drop1')(o)
    o = Conv2D(filters=2048, kernel_size=(1, 1), activation='relu', padding='same', data_format='channels_last',
               name='increase_conv2')(o)
    o = Dropout(0.5, name='drop2')(o)
    # Predict layer
    o = Conv2D(n_classes, (1, 1), data_format='channels_last', name='pred_conv1')(o)

    # Deconv by 2 for adding.
    o = Conv2DTranspose(n_classes, kernel_size=(32, 32), strides=(2, 2), padding='same', use_bias=False,
                        data_format='channels_last', name='Deconv1')(o)
    # Features from f2 layer
    o1 = f2

    o1 = AveragePooling3D(pool_size=(2, 1, 1))(o1)

    o1 = Reshape((60, 80, 128))(o1)
    # print(o1.shape)
    ##############################
    # Respath block
    # o1 = ResPath(n_classes, 2, o1)
    o1 = Conv2D(n_classes, (1, 1), data_format='channels_last', name='pred_conv2')(o1)

    o = concatenate([o, o1], axis=3)
    o = Conv2D(n_classes, (1, 1), data_format='channels_last', name='concat1_conv')(o)
    # Deconv by 4->2
    o = Conv2DTranspose(n_classes, kernel_size=(32, 32), strides=(2, 2), padding='same', use_bias=False,
                        data_format='channels_last', name='Deconv2')(o)

    # feature from f1 layer
    o2 = f1
    # print(o2.shape)
    # o2 = Lambda(lambda x:x[:,-1,:,:,:], name='select2')(o2) #use last channel as our prediction is final frame's mask
    o2 = AveragePooling3D(pool_size=(4, 1, 1))(o2)
    # print(o2.shape)
    o2 = Reshape((120, 160, 64))(o2)
    o2 = Conv2D(n_classes, (1, 1), data_format='channels_last', name='pred_conv3')(o2)
    # sum them together
    o = concatenate([o, o2], axis=3)
    o = Conv2D(n_classes, (1, 1), data_format='channels_last', name='concat2_conv')(o)
    # Deconv by 2
    o = Conv2DTranspose(n_classes, kernel_size=(32, 32), strides=(2, 2), padding='same', use_bias=False,
                        data_format='channels_last', name='Deconv3')(o)

    # softmax
    o = Activation('softmax', name='softmax')(o)

    o_shape = Model(img_input, o).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o_flatten = Reshape((outputHeight * outputWidth, -1), name='reshape_out')(o)

    model = Model(img_input, o_flatten)
    return model