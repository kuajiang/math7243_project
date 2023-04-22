import ast
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from common import stat_shape
from common import IMAGE_SHAPE, RANDOM_SEED


def read_segmentation_data():
    
    table1 = pd.read_csv('./labels/Results_Intraparenchymal_Hemorrhage_Detection_2020-11-16_21.39.31.268.csv')
    table1['Correct Label'] = table1['Correct Label'].replace('[[], []]',np.nan)
    table1['Majority Label'] = table1['Majority Label'].replace('[]',np.nan)
    table1['hemorrhage_type'] = 'intraparenchymal'

    table2 = pd.read_csv('./labels/Results_Intraventricular_Hemorrhage_Tracing_2020-09-28_15.21.52.597.csv')
    table2['Correct Label'] = table2['All Annotations'].replace('[[], []]',np.nan)
    table2['Majority Label'] = table2['All Annotations'].replace('[]',np.nan)
    table2['hemorrhage_type'] = 'intraventricular'

    table3 = pd.read_csv('./labels/Results_Subarachnoid_Hemorrhage_Detection_2020-11-16_21.36.18.668.csv')
    table3['Correct Label'] = table3['Correct Label'].replace('[[], []]',np.nan)
    table3['Majority Label'] = table3['Majority Label'].replace('[]',np.nan)
    table3['hemorrhage_type'] = 'subarachnoid'

    table4 = pd.read_csv('./labels/Results_Subdural_Hemorrhage_Detection_2020-11-16_21.35.48.040.csv')
    table4['Correct Label'] = table4['Correct Label'].replace('[[]]',np.nan)
    table4['Majority Label'] = table4['Majority Label'].replace('[]',np.nan)
    table4['hemorrhage_type'] = 'subdural'

    table5 = pd.read_csv('./labels/Results_Epidural_Hemorrhage_Detection_2020-11-16_21.31.26.148.csv')
    table5['Correct Label'] = table5['Correct Label'].replace('[[], []]',np.nan)
    table5['Majority Label'] = table5['Majority Label'].replace('[]',np.nan)
    table5['hemorrhage_type'] = 'epidural'

    table6 = pd.read_csv('./labels/Results_Multiple_Hemorrhage_Detection_2020-11-16_21.36.24.018.csv')
    table6['Correct Label'] = table6['Correct Label'].replace('[[], []]',np.nan)
    table6['Majority Label'] = table6['Majority Label'].replace('[]',np.nan)
    table6['hemorrhage_type'] = 'multi'

    tables = dict()
    tables['intraparenchymal'] = table1
    tables['intraventricular'] = table2
    tables['subarachnoid'] = table3
    tables['subdural'] = table4
    tables['epidural'] = table5
    tables['multi'] = table6

    return tables

def label_arr(points):
    if len(points)==0:
        return np.full((512,512), False,dtype=bool)
    xlabel = [p['x']*512 for p in points]
    ylabel = [p['y']*512 for p in points]
    img = Image.new('1', (512,512),color=0)
    draw = ImageDraw.Draw(img)
    xy = list(zip(xlabel, ylabel))
    draw.polygon(xy,fill=1)
    arr = np.array(img)   
    return arr

def table_label(table,hemorrhage_type):
    for idx, row in table.iterrows():
        pic = np.full((512,512), False,dtype=bool)
        row = table.loc[idx]
        # print(row)
        pics =  table[table['Origin']==table['Origin'][idx]]
        correct = pics['Correct Label'].tolist()
        for i in range(len(pics)):
            if type(correct[i])==type(np.nan):
                string = pics['Majority Label'].tolist()[i]
                if type(string)==type(np.nan):
                    img_pil = Image.fromarray(pic.astype(np.uint8) * 255)
                    img_pil.save('./seg-label/'+hemorrhage_type+'/label_'+table['Origin'][idx][:-4]+'.png')
                    return None
                string = string.replace("[]","").replace('\'\'',"")
            else:
                string = correct[i].replace("[]","").replace('\'\'',"")
            list_pattern = r'\[[^\[\]]*\]'
            valid_lists = re.findall(list_pattern, string)
            circles = [x for x in map(ast.literal_eval, valid_lists) if isinstance(x, list)]         
            for points in circles:
                pic+=label_arr(points)
        img_pil = Image.fromarray(pic.astype(np.uint8) * 255)
        img_pil.save('./seg-label/'+hemorrhage_type+'/label_'+table['Origin'][idx][:-4]+'.png')
    return None


def unet_mobilenetv2(input_shape=IMAGE_SHAPE, num_classes=1):
    """ Construct a U-Net model with MobileNetV2 encoder

    Args:
        input_shape: Input image shape
        num_classes: Number of classes
    """
    # Load pre-trained MobileNetV2 as the encoder
    encoder = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Encoder layers
    s1 = encoder.get_layer('block_1_expand_relu').output
    s2 = encoder.get_layer('block_3_expand_relu').output
    s3 = encoder.get_layer('block_6_expand_relu').output
    s4 = encoder.get_layer('block_13_expand_relu').output
    encoder_output = encoder.get_layer('block_16_project').output

    # Decoder layers
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(encoder_output)
    x = tf.keras.layers.Concatenate()([x, s4])

    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s3])

    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s2])

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s1])

    x = tf.keras.layers.Conv2DTranspose(num_classes, (3, 3), strides=(2, 2), padding='same')(x)

    # Output layer
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = tf.keras.layers.Activation(activation)(x)

    # Construct the U-Net model with MobileNetV2 encoder
    model = tf.keras.Model(inputs=encoder.input, outputs=output)

    return model


class DataGenerator(Sequence):
    def __init__(self, df, batch_size=8, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indexes = self.indexes[start:end]

        batch_df = self.df.iloc[indexes]
        
        batch_images = []
        batch_masks = []
        for _, row in batch_df.iterrows():
            image_file = "./renders/%s/brain_window/%s.jpg" % (row['hemorrhage_type'], row['image'])
            mask_file = "./seg-label/%s/%s" % (row['hemorrhage_type'], row['mask'])
            image = load_image(image_file)
            mask = read_mask_file(mask_file)

            batch_images.append(image)
            batch_masks.append(mask)

        X = np.array(batch_images, dtype=np.float32)
        y = np.array(batch_masks, dtype=np.float32)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)



def load_image(path, img_size=IMAGE_SHAPE):
    img = tf.keras.preprocessing.image.load_img(path, target_size=img_size[0:2])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array/255.0



def read_mask_file(mask_file, target_size=IMAGE_SHAPE):
    image = Image.open(mask_file)

    # Convert the image to grayscale (black and white)
    image_gray = image.convert('L')

    # Resize the image to 256x256
    image_resized = image_gray.resize(target_size[0:2], Image.LANCZOS)

    # Convert the image to a NumPy array and normalize the values
    image_array = np.array(image_resized) / 255.0

    # Add the channel dimension (1) to the array
    image_array = image_array.reshape(IMAGE_SHAPE[0:2] + (1,))

    return image_array


def load_image_mask_by_table(table):
    all_images = []
    all_masks = []
    for index, row in table.iterrows():
        image_file = "./renders/%s/brain_window/%s.jpg" % (row['hemorrhage_type'], row['image'])
        mask_file = "./seg-label/%s/%s" % (row['hemorrhage_type'], row['mask'])
        image = load_image(image_file)
        mask = read_mask_file(mask_file)

        all_images.append(image)
        all_masks.append(mask)
    
    return all_images, all_masks

def prepare_mask_table():
    """Prepare a table with image, mask, hemorrhage type and shape

    read from ./labels/hemorrhage-labels-shape.csv
    read mask file names from ./seg-label/
    save to ./labels/hemorrhage-labels-mask.csv
    """
    table_shape = pd.read_csv('./labels/hemorrhage-labels-shape.csv')

    table = pd.DataFrame(columns=["image", "mask", "hemorrhage_type", "shape"])

    for hemorrhage_type in ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "multi"]:
        mask_path = "./seg-label/%s/" % hemorrhage_type
        for file in os.listdir(mask_path):

            image_name = file[6:-4]
            image_file = "./renders/%s/brain_window/%s.jpg" % (hemorrhage_type, image_name)

            if not os.path.exists(image_file):
                print("Image file does not exist:", image_file)
                continue

            # find shape
            shape = table_shape.loc[table_shape['Image'] == image_name]['shape'].values[0]

            record = {
                "image": image_name,
                "mask": file,
                "hemorrhage_type": hemorrhage_type,
                "shape": shape
            }
            # print(record)
            
            # append record to table
            table = pd.concat([table, pd.DataFrame(record, index=[0])], ignore_index=True)
    table.to_csv("./labels/hemorrhage-labels-mask.csv", index=False)

def metric_fun(y_true,y_pred):
    fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
    fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
    return fz / fm

def train_model_seg():
    # Load the data
    df = pd.read_csv('./labels/hemorrhage-labels-mask.csv')
    print("total records: ", len(df))
    stat_shape(df)

    # Split the data
    test_df = df.sample(frac=0.2, random_state=RANDOM_SEED)
    remain_df = df.drop(test_df.index)
    train_df = remain_df.sample(frac=0.8, random_state=RANDOM_SEED)
    valid_df = remain_df.drop(train_df.index)

    # Create the U-Net model with MobileNetV2 encoder
    model = unet_mobilenetv2(input_shape=IMAGE_SHAPE, num_classes=1)
    # model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
              metrics=[metric_fun, 'accuracy'])

    # Train the model

    # Create data generators for training and validation data
    train_generator = DataGenerator(train_df, batch_size=64)
    val_generator = DataGenerator(valid_df, batch_size=64)

    early_stopping_callback = EarlyStopping(monitor='metric_fun', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('./models/seg_weights_epoch_{epoch:02d}.h5', save_weights_only=True)



    # Train the model using the data generators
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=val_generator,
        callbacks=[early_stopping_callback, checkpoint]
    )

    # Save the model:
    model.save('./models/seg_weight.h5')
    # Save the history:
    with open('./models/seg_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return model, history


def evaluate_model_seg():
    model = unet_mobilenetv2(input_shape=IMAGE_SHAPE, num_classes=1)
    model.load_weights('./models/seg_unet_1550_59_weight.h5')


    # Load the data
    table = pd.read_csv('./labels/hemorrhage-labels-mask.csv')
    samples = table.sample(n=1)
    images, masks = load_image_mask_by_table(samples)

    images = np.array(images)
    masks = np.array(masks)

    print(images.shape, masks.shape)

    # Predict
    y_pred = model.predict(images)

    # Plot the results
    for i in range(len(images)):
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(images[i]/255.0)

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(masks[i])

        plt.subplot(1, 3, 3)
        plt.title("Predicted mask")
        plt.imshow(y_pred[i])

        plt.show()


def gen_mask():
    print("read labels")
    labels = read_segmentation_data()
    print("generate mask images")
    for hemorrhage_type, df in labels.items():
        print(hemorrhage_type, len(df))
        # ensure dir exists
        if not os.path.exists("./seg-label/%s" % hemorrhage_type):
            os.makedirs("./seg-label/%s" % hemorrhage_type)
        table_label(df, hemorrhage_type)


def main():
    # generate mask images
    # gen_mask()

    # generate csvn file for mast images, only for convenience
    # prepare_mask_table()

    # train the segmentation model
    # train_model_seg()

    # evaluate the segmentation model with test data, random select one image
    evaluate_model_seg()


if __name__ == '__main__':
    main()