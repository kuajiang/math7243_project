import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.image as mpimg
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16



from sklearn.model_selection import train_test_split
import time
import os
from typing import List


def load_cleaning_filenames():
    # read all file names to a set
    file_name_type = dict()

    for i in range(0, 5):
        dir_path = './cleaning/type%d' % i
        for file in os.listdir(dir_path):
            file_name_type[file] = i
    print("total file names: ", len(file_name_type))
    return file_name_type

def save_cleaning_csv(table, file_names):
    # save to csv
    table['filename'] = table['Image'].apply(lambda x: x + '.jpg')
    table = table[table['filename'].isin(file_names)]
    table['shape_type'] = table['filename'].apply(lambda x: file_names[x])
    table.to_csv('./labels/hemorrhage-labels-cleaning.csv', index=False)
    print("saved to csv")

    hemorrhage_types = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural', 'normal']
    print(table[hemorrhage_types].sum(axis=0))


def load_cleaning_data(table, img_size = (128, 128)):
    selected_columns = ["shape_type"]

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



def build_multiclass_cnn(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Use softmax activation for multi-class classification

    return model

def build_fine_tuned_cnn(input_shape, num_classes):
    # Load the pre-trained VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers to prevent them from updating during training
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers for classification
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def train_model(train_data, train_labels, validation_data, validation_labels, test_images, test_labels):
    
    input_shape = train_data.shape[1:]
    num_classes = train_labels.shape[1]
    # model = build_multiclass_cnn(input_shape, num_classes)
    model = build_fine_tuned_cnn(input_shape, num_classes)
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = 5
    batch_size = 64
    
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))
    
    # Evaluate the model:
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print('Test loss: {}, test accuracy: {}'.format(test_loss, test_accuracy))

    # save model
    model.save('./models/cleaning.h5')
    return history

def main():
    # table = pd.read_csv('./labels/hemorrhage-labels-exist.csv')
    # filenames = load_cleaning_filenames()
    # save_cleaning_csv(table, filenames)

    # exit()
    table = pd.read_csv('./labels/hemorrhage-labels-cleaning.csv')
    VGG16_EXPECT_IMAGE_SIZE = (224, 224)
    IMAGE_SIZE = (256, 256)
    images, labels = load_cleaning_data(table, img_size=VGG16_EXPECT_IMAGE_SIZE)
    
    print(labels)

    labels_one_hot = to_categorical(np.array(labels), num_classes=5)
    print(labels_one_hot)
    
    print(images.shape)
    print(labels_one_hot.shape)

    
    # train model
    train_ratio = 0.7
    test_ratio = 0.2
    val_ratio = 0.1    

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels_one_hot, test_size=(test_ratio + val_ratio))
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(val_ratio / (test_ratio + val_ratio)))
    print(y_train.shape)

    train_model(X_train, y_train, X_val, y_val, X_test, y_test)



if __name__ == '__main__':
    main()