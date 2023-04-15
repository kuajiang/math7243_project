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

VGG16_EXPECT_IMAGE_SIZE = (224, 224)


from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array



def update_shape_table():
    # check file exists
    if not os.path.exists('./labels/hemorrhage-labels-shape.csv'):
        table = pd.read_csv('./labels/hemorrhage-labels-exist.csv')
        table["shape"] = 9

        table.to_csv('./labels/hemorrhage-labels-shape.csv', index=False)

    table = pd.read_csv('./labels/hemorrhage-labels-shape.csv')

    
    # load model
    model_path = './models/cleaning.h5'
    loaded_model = tf.keras.models.load_model(model_path)


    for i in range(1000):
        batch_start_time = time.time()
        table_unshaped = table[table['shape'] == 9]
        table_select = table_unshaped.sample(n=32)
        
        images = []

        for _, row in table_select.iterrows():
            img_file = './renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image'])

            input_image = preprocess_image(img_file, target_size=VGG16_EXPECT_IMAGE_SIZE)

            images.append(input_image)

        images = np.vstack(images)
        predictions = loaded_model.predict(images)
        predicted_class = np.argmax(predictions, axis=-1)
        print(predicted_class)

        count = 0
        for idx, row in table_select.iterrows():
            # print(idx, predicted_class[count], row.Image)
            table.loc[idx, 'shape'] = predicted_class[count]
            count += 1
        
        batcch_end_time = time.time()
        print('batch time: ', batcch_end_time - batch_start_time)
    
        table.to_csv('./labels/hemorrhage-labels-shape.csv', index=False)

    # images = images * 255
    # images = np.array(images, dtype=np.uint8)
    # plot_images(images, predicted_class, nrows=2, ncols=5)




import matplotlib.pyplot as plt

def plot_images(images, labels, nrows=2, ncols=5):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.ravel()

    for i in range(nrows * ncols):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()




def main():
    update_shape_table()
    # table = pd.read_csv('./labels/hemorrhage-labels-exist.csv')
    # filenames = load_cleaning_filenames()
    # save_cleaning_csv(table, filenames)

    # load model
    # model = tf.keras.models.load_model('./models/cleaning.h5')



if __name__ == '__main__':
    main()