#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 2022

@author: shan
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms

n_classes = 2
input_shape = X_train.shape[1:]
img_input = Input(shape=(input_shape), name='input')
kernel_size = (3, 3, 3)
pool_size = (2, 2, 2)
# kernel_size = (3,3)
# pool_size = (2,2)

# strides_size = (2,2,2)
filter_size = 64


# In[10]:

# In[10]:
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


def make_model(input_shape, n_classes, filter_size, kernel_size, pool_size):
    # This code structure is based on Image Segmentation Keras by divamgupta.
    # https://github.com/divamgupta/image-segmentation-keras

    # Block1
    res_block = ResBlock(filter_size, part1)
    conv3d = conv3d_bn( filter_size, 3, 3, 3, padding='same', strides=(1, 1, 1), activation='relu', name=None)
    bn =
    # Block1
    part1, part2 = torch.split(img_input, 2, 1) # bschw->b 2 s/2 chw
    part1 = ResBlock(filter_size, part1)
    part2 = conv3d_bn(part2, filter_size, 3, 3, 3, padding='same', strides=(1, 1, 1), activation='relu', name=None)
    x = torch.cat([part1, part2], dim=1)
    x = torch.group_norm(x,32)
    # x = BatchNormalization(name='bn1')(x)
    x = torch.max_pool3d(x,pool_size)
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

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, kernel_size=3, conv_channel=64):
        self.conv3d = nn.conv3d(in_ch, conv_channel, kernel_size, padding='same', bias=False)
        self.norm = nn.GroupNorm(32,conv_channel) # 64c to 32 group
        self.active = nn.relu()
    def forward(self, x, active=True):
        x = self.conv3d(x)
        x = self.norm(x)
        if active:
            x = self.active(x)
        return x

class ResBlock(nn.Module):
    def __init__(self):
        self.conv3d_1 = Conv3DBlock(3,kernel_size=1, conv_channel=64)
        self.conv3d_2 = Conv3DBlock(64, kernel_size=3, conv_channel=64)
        self.conv3d_3 = Conv3DBlock(64, kernel_size=3, conv_channel=64)
        self.norm = nn.GroupNorm(32,64) # 64c to 32 group
        self.active = nn.relu()
    def forward(self, x):
        x = self.conv3d_1(x, active=False)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.norm(x)
        x = self.active(x)
        x = self.norm(x)

        return x

class ResPath(nn.Module):
    def __init__(self, middle_ch):
        self.conv3d_1 = Conv3DBlock(64, middle_ch, kernel_size=1, active=False)
        self.conv3d_3 = Conv3DBlock(64, middle_ch, kernel_size=3)
        self.norm = nn.GroupNorm(32,64) # 64c to 32 group
        self.active = nn.relu()
    def forward(self, x, length):
        shortcut = self.conv3d_1(x)
        x = self.conv3d_3(x)
        x = torch.cat([shortcut,x], dim=2) # [b,s,c,h,w]
        x = self.active(x)
        x = self.norm(x)
        for i in range(length-1):
            shortcut = self.conv3d_1(x)
            x = self.conv3d_3(x)




class Conv3DNet(nn.Module):
    def __init__(self, sequence=16):
        super(Net, self).__init__()
        self.sequence = sequence
        self.res_block = ResBlock()
        self.conv3d_block = Conv3DBlock(3,kernel_size=3, conv_channel=64)
        self.norm = nn.GroupNorm(32, 64)
        self.conv2d = nn.Conv2d(64,2048,7)


    def forward(self, left, light):
        # left,right from current to previous #[b,s,c,h,w]
        # debug, save inputs
        if 0:#debug
            for i in range(left.size(1)):
                img = transforms.functional.to_pil_image(left[0,i],mode='RGB')
                img.save('input_left_'+str(i)+'.png')
            img = transforms.functional.to_pil_image(light[0, -1], mode='RGB')
            img.save('input_right.png')

        # Block1
        part1,part2 = torch.split(left, 2, 1) #[b,s/2,c,h,w]
        part1 = self.res_block(part1)
        part2 = self.conv3d_block(part2)
        left = torch.cat([part1,part2],dim=1)
        left = self.norm(left)
        left = torch.max_pool3d(left,2)

        part1, part2 = torch.split(left, 2, 1)
        part1_f = self.ResPath()
        part2_f = self.conv3d_block(part2)
        f1 = torch.cat([part1_f,part2_f],dim=1)

        # Block 2 # only res_block on part1
        part1 = self.res_block(part1)
        left = torch.cat([part1, part2], dim=1)
        left = self.norm(left)
        left = torch.max_pool3d(left,2)

        part1, part2 = torch.split(left, 2, 1)
        part1_f = self.ResPath()
        part2_f = self.conv3d_block(part2)
        f2 = torch.cat([part1_f,part2_f],dim=1)

        # Block 3 # only res_block on part1
        part1 = self.res_block(part1)
        left = torch.cat([part1, part2], dim=1)
        left = self.norm(left)
        left = torch.max_pool3d(left, 2)
        f3 = left #[b,s,c,h,w]

        # squeeze 0??
        left =


        # only left
        prd_mask = self.reflec_detect(left.flatten(1,2))

        # both left and right
        # both = torch.cat([left,right],dim=1) #[b,2s,c,h,w]
        # prd_mask = self.reflec_detect(both.flatten(1, 2))

        prd_mask = torch.sigmoid(prd_mask.squeeze(1))

        if 0:# debug
            img = transforms.functional.to_pil_image((prd_mask[0]>=0.5).float(),mode='L')
            #img = transforms.functional.to_pil_image(prd_mask[0], mode='L')
            img.save('predict_mask.png')

        return prd_mask
    