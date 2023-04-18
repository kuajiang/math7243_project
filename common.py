
import numpy as np
import pandas as pd
import tensorflow as tf

RANDOM_SEED = 255
ALL_HEMORRHAGE_TYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
ALL_IMAGE_DIRS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "normal", "multi"]
IMAGE_SHAPE = (224, 224, 3)

def load_image_data_by_df(df, input_shape, label_fields):
    images = np.zeros((len(df), *input_shape))
    labels = np.zeros((len(df), len(label_fields)))
    
    for index, (_, row) in enumerate(df.iterrows()):
        image_file = "./renders/%s/brain_window/%s.jpg" % (row['hemorrhage_type'], row['Image'])
        # print(image_file)
        
        img = tf.keras.preprocessing.image.load_img(image_file, target_size=input_shape[0:2])
        img_array = tf.keras.preprocessing.image.img_to_array(img)


        images[index] = img_array/255.0
        labels[index] = row[label_fields]

    # print("load images and labels", images.shape, labels.shape)
    # print(labels)
    # print(df.head(10))
    return images, labels

def select_for_hemorrhage_type(df, hemorrhage_type:str, count:int=0, random_state=255):
    # filter by shape == 2
    # df = df[df['shape'] == 2]

    # filter by target type
    filtered_df = df[df['hemorrhage_type'] == hemorrhage_type]

    if count == 0:
        count = len(filtered_df)
    else:
        if count > len(filtered_df):
            count = len(filtered_df)
    
    df_by_types = []

    for current_type in ALL_IMAGE_DIRS:
        filtered_df = df[df['hemorrhage_type'] == current_type]
        current_count = count
        if current_type == 'multi' or current_type in ALL_HEMORRHAGE_TYPES and current_type != hemorrhage_type:
            current_count = count // 5
        if current_count > len(filtered_df):
            current_count = len(filtered_df)
        sampled_df = filtered_df.sample(n=current_count, random_state=random_state)
        df_by_types.append(sampled_df)
    
    # normal_df = df[df['hemorrhage_type'] == 'normal']
    # current_count = count
    # if current_count > len(normal_df):
    #     current_count = len(normal_df)
    # sampled_df = normal_df.sample(n=current_count, random_state=random_state)
    # df_by_types.append(sampled_df)

    df = pd.concat(df_by_types)
    count_table = pd.crosstab(df['hemorrhage_type'], df['shape'])
    print(count_table)
    return df

def select_for_hemorrhage(df, count:int=0, random_state=255):
    df_normal = df[df['hemorrhage_type'] == 'normal']
    df_hemorrhage = df[df['hemorrhage_type'] != 'normal']

    count_normal = count
    if count_normal > len(df_normal):
        count_normal = len(df_normal)
    count_hemorrhage = count
    if count_hemorrhage > len(df_hemorrhage):
        count_hemorrhage = len(df_hemorrhage)
    
    df_normal = df_normal.sample(n=count_normal, random_state=random_state)
    df_hemorrhage = df_hemorrhage.sample(n=count_hemorrhage, random_state=random_state)

    df = pd.concat([df_normal, df_hemorrhage])
    count_table = pd.crosstab(df['hemorrhage_type'], df['shape'])
    print(count_table)
    return df

def filter_df_by_seglabel(df):
    df_seglabel = pd.read_csv('./labels/hemorrhage-labels-mask.csv')

    # get image values with shape in (1,2,3)
    df_seglabel = df_seglabel[df_seglabel['shape'].isin([1,2,3])]
    # convert image values to a set
    image_set = set(df_seglabel['image'].values)

    # filter df by image_set
    df = df[df['Image'].isin(image_set)]
    print("filter_df_by_seglabel", len(df))

    return df

def select_by_shape(df, shapes=[2]):
    df = df[df['shape'].isin(shapes)]
    return df

def select_average_type(df, count):
    df_for_types = []
    for hemorrhage_type in ALL_HEMORRHAGE_TYPES + ['multi']:
        df_type = df[df['hemorrhage_type'] == hemorrhage_type]
        current_count = count
        if current_count>len(df_type):
            current_count = len(df_type)
        df_type = df_type.sample(n=count, random_state=RANDOM_SEED)
        df_for_types.append(df_type)

    df_normal = df[df['hemorrhage_type'] == 'normal']
    count_normal = count * 5
    if count_normal > len(df_normal):
        count_normal = len(df_normal)
    df_normal = df_normal.sample(count_normal, random_state=RANDOM_SEED)
    df_for_types.append(df_normal)

    df = pd.concat(df_for_types)
    return df



def stat_shape(df):
    count_table = pd.crosstab(df['hemorrhage_type'], df['shape'])
    print(count_table)




from sklearn.model_selection import train_test_split
def split_df(df, train_ratio=0.7, val_ratio=0.2, random_state=255):
    test_ratio = 1 - train_ratio - val_ratio
    train_df, validate_test_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    test_df, validate_df = train_test_split(validate_test_df, test_size=(val_ratio / (test_ratio + val_ratio)))

    return train_df, validate_df, test_df


import time
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