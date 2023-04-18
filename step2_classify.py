import os
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from common import load_image_data_by_df, stat_shape, select_by_shape
from common import IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES, RANDOM_SEED

from build_models import multilabel_mobilenet, multilabel_inception

import pickle


# Create a custom data generator using Keras' Sequence class
class ClassifyDataGenerator(Sequence):
    def __init__(self, df, batch_size, input_shape, label_fields):
        self.df = df
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_fields = label_fields

    def __len__(self):
        # print("get len", int(np.ceil(len(self.df) / float(self.batch_size))))
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        return load_image_data_by_df(batch_df, self.input_shape, self.label_fields)


def train_model_by_flow(model, train_df, valid_df, label_fields):
    batch_size = 64

    # print("train_df", len(train_df))
    # print("valid_df", len(valid_df))

    train_generator = ClassifyDataGenerator(train_df, batch_size, IMAGE_SHAPE, label_fields)
    valid_generator = ClassifyDataGenerator(valid_df, batch_size, IMAGE_SHAPE, label_fields)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create an EarlyStopping callback that monitors the validation accuracy
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)


    # Train the model using the custom data generators
    num_epochs = 3  # Set the number of epochs as needed
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=num_epochs,
        callbacks=[early_stopping_callback]
    )

    # Save the model:
    model_file_name = './models/classify-model-%s.h5' % int(time.time())

    model.save(model_file_name)
    # Save the history:
    history_file_name = './models/classify-history-%s.pkl' % int(time.time())
    with open(history_file_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return history

def main():
    df = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    print("total records: ", len(df))
    stat_shape(df)

    # select by shape
    # df = df[df['shape'].isin([1,2,3])]


    # select all
    df_after_select = df
    df_after_select = df.sample(100, random_state=RANDOM_SEED)


    # split into training and validation sets
    test_df = df_after_select.sample(frac=0.2, random_state=RANDOM_SEED)
    remain_df = df_after_select.drop(test_df.index)
    train_df = remain_df.sample(frac=0.8, random_state=RANDOM_SEED)
    valid_df = remain_df.drop(train_df.index)

    # build multilabel classification model
    model = multilabel_mobilenet(len(ALL_HEMORRHAGE_TYPES))
    model = multilabel_inception(len(ALL_HEMORRHAGE_TYPES))

    # train model
    train_model_by_flow(model, train_df, valid_df, ALL_HEMORRHAGE_TYPES)

if __name__ == '__main__':
    main()