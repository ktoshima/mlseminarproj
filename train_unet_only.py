from data import train_data_generator, stack_gen
from model_bn import unet
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from os import makedirs
import tensorflow as tf
import pandas as pd

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
    model_save_path = parent_dir + "/unet_only{seed}.hdf5".format(seed=seed)
    train_df = pd.read_csv(parent_dir+"/train_data_seed{seed}.csv".format(seed=seed), index_col=['timestamp', 'num'], parse_dates=['timestamp'])

    band1_gen, band3_gen, mask_gen, val_band1_gen, val_band3_gen, val_mask_gen = train_data_generator(
        dataframe=train_df,
        image_side_length=image_side_length,
        batch_size=batch_size,
        seed=seed,
        validation_split_rate=validation_split_rate,
        classify_level='density'
        )

    # build model
    model = unet(
        input_size=(image_side_length, image_side_length, 2),
        # pretrained_weights=model_load_path,
        learning_rate=3e-4,
        classify_level=4,
        auto_encoder=False
        )

    # save weights with least loss function value per epoch
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)

    # early stopping
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, mode='min', verbose=1, patience=3)

    # train!
    history = model.fit(
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
    hist_df.to_csv(parent_dir+'/unet_only_history.csv')
