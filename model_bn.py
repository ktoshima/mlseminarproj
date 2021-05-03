from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import *
from keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalCrossentropy, MeanSquaredError
import keras.backend as K
import tensorflow as tf
from keras.objectives import mean_squared_error

def unet(input_size, pretrained_weights=None, learning_rate=1e-4, auto_encoder=False, classify_level=4):
    """
    Model based on Unet (https://arxiv.org/abs/1505.04597) with slight modification with categorization/loss function
    Args:
    - input_size: shape of each input image (side length, side length, channel)
    - pretrained_weights: path to pretrained weights, if any
    - learning_rate: learning rate for Adam optimizer
    - classify_level: integer of how many level for HMS. 4 for density and 2 for binary.
    """

    initializer = 'he_normal'

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv9 = BatchNormalization()(conv9)
    if auto_encoder:
        conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)
        loss_func = 'mean_squared_error'
        metrics = [
            MeanSquaredError()
            ]
    else:
        conv10 = Conv2D(classify_level, 1, activation = 'softmax')(conv9)
        loss_func = 'sparse_categorical_crossentropy'
        metrics = [
            SparseTopKCategoricalAccuracy(k=1),
            SparseCategoricalCrossentropy(axis=-1)
            ]

    model = Model(inputs = inputs, outputs = conv10)

    # loss_func = lambda y_true, y_pred: jaccard_distance(y_true, y_pred, classify_level=classify_level, smooth=100)


    model.compile(
        optimizer = Adam(lr = learning_rate),
        loss = loss_func,
        metrics = metrics
        )

    if pretrained_weights:
    	model.load_weights(pretrained_weights)

    return model

def mse(y_true, y_pred):
    mses = mean_squared_error(y_true, y_pred)
    return K.sum(mses, axis=(1,2))


def jaccard_distance(y_true, y_pred, classify_level=4, smooth=1e-6):
    #flatten label and prediction tensors
    y_true = tf.one_hot(indices=tf.cast(y_true, tf.uint8), depth=classify_level)
    inputs = K.batch_flatten(y_pred)
    targets = K.batch_flatten(y_true)

    intersection = K.sum(targets * inputs, axis=-1, keepdims=True)
    total = K.sum(targets, axis=-1, keepdims=True) + K.sum(inputs, axis=-1, keepdims=True)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

    # y_true = tf.one_hot(indices=y_true, depth=classify_level)
    # intersection = keras.sum(y_true * y_pred, axis=-1)
    # sum_ = keras.sum(y_true + y_pred, axis=-1)
    # jac = (intersection + smooth) / (sum_ - intersection + smooth)
    # return (1 - jac) * smooth
