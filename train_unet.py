from data import train_data_generator, stack_gen
from model_bn import unet
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from os import makedirs
import tensorflow as tf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from random import randint

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import *
from keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalCrossentropy, MeanSquaredError
import keras.backend as K
import tensorflow as tf
from keras.objectives import mean_squared_error


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.compat.v1.disable_eager_execution()

    image_side_length = 512
    batch_size = 4
#     seed = randint(0, 100)
    seed = 49
    print("Seed is {seed}".format(seed=seed))
    validation_split_rate = 0.1
    epoch_num = 30
    parent_dir = "transfer_learning_noes"
    model_save_path = parent_dir+"/unet{seed}.hdf5".format(seed=seed)
    model_load_path = parent_dir+"/autoencoder{seed}.hdf5".format(seed=seed)
    train_df = pd.read_csv(parent_dir+"/train_data_seed{seed}.csv".format(seed=seed), index_col=['timestamp', 'num'], parse_dates=['timestamp'])


    # build model
    auto_encoder_model = unet(
        input_size=(image_side_length, image_side_length, 2),
        pretrained_weights=model_load_path,
        learning_rate=3e-4,
        classify_level=4,
        auto_encoder=True
        )

    conv_outputs = {}
    for i in range(1, 6):
        l_idx = 3 + 4*(i-1)
        conv_outputs[i] = auto_encoder_model.layers[l_idx].output

    initializer = 'he_normal'
    classify_level = 4

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_outputs[5]))
    merge6 = concatenate([conv_outputs[4] ,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv_outputs[3], up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv_outputs[2], up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv_outputs[1], up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(classify_level, 1, activation = 'softmax')(conv9)
    loss_func = 'sparse_categorical_crossentropy'
    metrics = [
        SparseTopKCategoricalAccuracy(k=1),
        SparseCategoricalCrossentropy(axis=-1)
        ]

    unet_model = Model(inputs=auto_encoder_model.input, outputs=conv10)
    for i in range(0, 20):
        unet_model.layers[i].trainable = False
    learning_rate = 3e-4
    unet_model.compile(
        optimizer = Adam(lr = learning_rate),
        loss = loss_func,
        metrics = metrics
        )

    band1_gen, band3_gen, mask_gen, val_band1_gen, val_band3_gen, val_mask_gen = train_data_generator(
        dataframe=train_df,
        image_side_length=image_side_length,
        batch_size=batch_size,
        seed=seed,
        validation_split_rate=validation_split_rate,
        classify_level='density'
        )

    # save weights with least loss function value per epoch
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)

    # early stopping
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, mode='min', verbose=1, patience=3)

    # train!
    history = unet_model.fit(
        x=stack_gen(band1_gen, band3_gen, mask_gen, auto_encoder=False),
        epochs=epoch_num,
        verbose=1,
        steps_per_epoch=len(mask_gen),
        validation_data=stack_gen(val_band1_gen, val_band3_gen, val_mask_gen, auto_encoder=False),
        validation_steps=len(val_mask_gen),
        callbacks=[
            model_checkpoint, 
#             early_stopping
        ]
        )
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(parent_dir+'/unet_history.csv')
