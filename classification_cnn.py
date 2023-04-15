import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.image as mpimg
from tensorflow.keras import layers, models

import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0


from sklearn.model_selection import train_test_split
import time

from typing import List

EFFICIENTNET_EXPECT_IMAGE_SIZE = (224, 224)



def select_records(df, count:int, select_type:str, random_state=255):
    all_types = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','normal']

    if select_type:
        print("all types count before selection")
        print(df[all_types].sum(axis=0))

        # init empty dataframe
        df_by_types = []

        for current_type in all_types:
            # filter by current_type
            filtered_df = df[df[current_type] == 1]
            print("count of current type: ", current_type, " is ", len(filtered_df))
            
            # sample
            select_count = count
            if current_type != select_type and current_type != 'normal':
                select_count = count // 4
            if select_count > len(filtered_df):
                select_count = len(filtered_df)
            sampled_df = filtered_df.sample(n=select_count, random_state=random_state)
            
            df_by_types.append(sampled_df)

        df_after_filter = pd.concat(df_by_types)
    else:
        # sample count rows
        df_after_filter = df.sample(n=count, random_state=random_state)

    if select_type:
        print("after selection: ", select_type, len(df_after_filter))
        print(df_after_filter[all_types].sum(axis=0))

    # shuffle
    df_after_shuffle = df_after_filter.sample(frac=1, random_state=random_state)

    return df_after_shuffle

def load_jpg_files_one_type(table, hemorrhage_type, img_size = EFFICIENTNET_EXPECT_IMAGE_SIZE):
    selected_columns = [hemorrhage_type]

    # read images and labels
    images = []
    labels = []
    
    start_time = time.time()  # Log the start time
    counter = 0
    for _, row in table.iterrows():
        img_file = './renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image'])
        # print(img_file, img_size)
        img = tf.keras.preprocessing.image.load_img(img_file, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Convert the NumPy array to float32 data type
        img_array = img_array.astype('float32')
        
        images.append(img_array)
        
        # images.append(img_array)
        labels.append(row[selected_columns].values)

        # Log the time used for loading every 1000 images
        counter += 1
        if counter % 100 == 0:
            end_time = time.time()  # Log the end time
            elapsed_time = end_time - start_time
            print(f"Time taken to load {counter} images: {elapsed_time:.2f} seconds")

    return np.array(images, dtype='float'), np.array(labels, dtype='int')

def load_jpg_files(table, img_size = (256, 256)):
    selected_columns = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','normal']

    # read images and labels
    images = []
    labels = []
    
    start_time = time.time()  # Log the start time
    counter = 0
    for index, row in table.iterrows():
        img_file = './renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image'])
        print(img_file, img_size)
        img = tf.keras.preprocessing.image.load_img(img_file, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Convert the NumPy array to float32 data type
        img_array = img_array.astype('float32')
        
        images.append(img_array)
        
        # images.append(img_array)
        labels.append(row[selected_columns].values)

        # Log the time used for loading every 1000 images
        counter += 1
        if counter % 100 == 0:
            end_time = time.time()  # Log the end time
            elapsed_time = end_time - start_time
            print(f"Time taken to load {counter} images: {elapsed_time:.2f} seconds")

    return np.array(images, dtype='float'), np.array(labels, dtype='int')







def train_efficientnet(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    # Load the pre-trained EfficientNet model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Create a custom classification head for the task
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification

    # Construct the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model on the dataset
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

    return model, history


import pickle
def train_for_hemo_type(table, hemorrhage_type):

    data = select_records(table, 1000, hemorrhage_type)
    
    # Load the data:
    images, labels = load_jpg_files_one_type(data, hemorrhage_type)
    print(images.shape)
    print(labels.shape)
    
    # Split the data
    train_ratio = 0.7
    test_ratio = 0.2
    val_ratio = 0.1    
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=(test_ratio + val_ratio))
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(val_ratio / (test_ratio + val_ratio)))

    # Train the model:
    model, history = train_efficientnet(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save the model:
    model.save('./models/efficientnet_%s.h5' % hemorrhage_type)

    # Save the history:
    with open('./models/efficientnet_%s_history.pkl' % hemorrhage_type, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return model, history



def train_model(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    # Build the model with the input shape and number of classes:
    input_shape = train_images[0].shape
    num_classes = 6

    # model = build_multilabel_cnn(input_shape, num_classes)

    model = build_onelabel_cnn(input_shape)

    # Compile the model:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model:
    history = model.fit(train_images, train_labels, batch_size=32, epochs=10,
                    validation_data=(val_images, val_labels))

    # Evaluate the model:
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print('Test loss: {}, test accuracy: {}'.format(test_loss, test_accuracy))


def main():
    table = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    print("total records: ", len(table))

    # select shape = 2
    table = table[table['shape'] == 2]
    print("total records with shape = 2: ", len(table))

    hemorrhage_types = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

    for hemorrhage_type in hemorrhage_types:
        if hemorrhage_type == 'epidural':
            continue
        train_for_hemo_type(table, hemorrhage_type)

if __name__ == '__main__':
    main()