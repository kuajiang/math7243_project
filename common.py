import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


RANDOM_SEED = 255
IMAGE_SHAPE = (224, 224, 3)
ALL_HEMORRHAGE_TYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
ALL_IMAGE_DIRS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "normal", "multi"]

def stat_shape(df):
    count_table = pd.crosstab(df['hemorrhage_type'], df['shape'])
    print(count_table)


def split_df(df, train_ratio=0.7, val_ratio=0.2, random_state=255):
    test_ratio = 1 - train_ratio - val_ratio
    train_df, validate_test_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    test_df, validate_df = train_test_split(validate_test_df, test_size=(val_ratio / (test_ratio + val_ratio)))

    return train_df, validate_df, test_df


def load_images(file_paths, img_size = IMAGE_SHAPE):

    # init images
    if len(img_size) == 2:
        img_size = img_size + (3,)
    images = np.zeros((len(file_paths), *img_size))
    
    start_time = time.time()  # Log the start time
    for idx, img_file in enumerate(file_paths):
        img = tf.keras.preprocessing.image.load_img(img_file, target_size=img_size[0:2])
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Convert the NumPy array to float32 data type
        img_array = img_array.astype('float32')
        
        images[idx] = img_array/255.0

        if idx % 100 == 0:
            end_time = time.time()  # Log the end time
            elapsed_time = end_time - start_time
            print(f"Time taken to load {idx} images: {elapsed_time:.2f} seconds")

    return images