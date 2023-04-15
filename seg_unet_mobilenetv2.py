import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import os


EXPECT_IMAGE_SIZE = (224, 224)
EXPECT_IMAGE_SHAPE = (224, 224, 3)


def unet_mobilenetv2(input_shape=EXPECT_IMAGE_SHAPE, num_classes=1):
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


def load_image(path, img_size=EXPECT_IMAGE_SIZE):

    img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array



def read_mask_file(mask_file, target_size=EXPECT_IMAGE_SIZE):
    image = Image.open(mask_file)

    # Convert the image to grayscale (black and white)
    image_gray = image.convert('L')

    # Resize the image to 256x256
    image_resized = image_gray.resize(target_size, Image.LANCZOS)

    # Convert the image to a NumPy array and normalize the values
    image_array = np.array(image_resized) / 255.0

    # Add the channel dimension (1) to the array
    image_array = image_array.reshape(EXPECT_IMAGE_SIZE + (1,))

    return image_array

def load_image_mask_by_type(hemorrage_type):
    mask_dir = "./seg-label/"
    mask_dir_by_type = mask_dir + hemorrage_type + "/"
    image_dir = "./renders/"+hemorrage_type+"/brain_window/"

    images = []
    masks = []

    for file in os.listdir(mask_dir_by_type):
        # print(file)
        image_file_name = file[6:-4]
        image_file_path = image_dir + image_file_name + ".jpg"
        # print(image_file_path)
        if not os.path.exists(image_file_path):
            print("Image file does not exist:", image_file_path)
            continue
        image = load_image(image_file_path)
        # print(image.shape)

        mask = read_mask_file(mask_dir_by_type + file)
        # print(mask.shape)

        images.append(image)
        masks.append(mask)
    
    return images, masks

def load_image_mask_all():
    all_images = []
    all_masks = []
    for hemorrage_type in ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]:
        images, masks = load_image_mask_by_type(hemorrage_type)
        all_images.extend(images)
        all_masks.extend(masks)
    
    return all_images, all_masks

def load_image_mask_by_table(table):
    all_images = []
    all_masks = []
    for index, row in table.iterrows():
        image_file = "./renders/%s/brain_window/%s.jpg" % (row['hemorrage_type'], row['image'])
        mask_file = "./seg-label/%s/%s" % (row['hemorrage_type'], row['mask'])
        image = load_image(image_file)
        mask = read_mask_file(mask_file)

        all_images.append(image)
        all_masks.append(mask)
    
    return all_images, all_masks

def prepare_mask_table():
    table_shape = pd.read_csv('./labels/hemorrhage-labels-shape.csv')

    table = pd.DataFrame(columns=["image", "mask", "hemorrage_type", "shape"])

    for hemorrage_type in ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]:
        mask_path = "./seg-label/%s/" % hemorrage_type
        for file in os.listdir(mask_path):
            mask_file = mask_path + file

            image_name = file[6:-4]
            image_file = "./renders/%s/brain_window/%s.jpg" % (hemorrage_type, image_name)

            if not os.path.exists(image_file):
                print("Image file does not exist:", image_file)
                continue

            # find shape
            shape = table_shape.loc[table_shape['Image'] == image_name]['shape'].values[0]

            record = {
                "image": image_name,
                "mask": file,
                "hemorrage_type": hemorrage_type,
                "shape": shape
            }
            print(record)
            
            # append record to table
            table = pd.concat([table, pd.DataFrame(record, index=[0])], ignore_index=True)
    table.to_csv("./labels/hemorrhage-labels-mask.csv", index=False)

def train_model_seg():
    # Load the data

    images, masks = load_image_mask_all()

    images = np.array(images)
    masks = np.array(masks)

    print(images.shape, masks.shape)

    # Split the data
    train_ratio = 0.7
    test_ratio = 0.2
    val_ratio = 0.1    
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, 
        masks, 
        test_size=(test_ratio + val_ratio), 
        random_state=255)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=(val_ratio / (test_ratio + val_ratio)),
        random_state=255)

    # Create the U-Net model with MobileNetV2 encoder
    model = unet_mobilenetv2(input_shape=EXPECT_IMAGE_SHAPE, num_classes=1)
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    # Train the model
    epochs = 20
    batch_size = 32

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size)

    # Save the model:
    model.save('./models/seg_unet.h5')
    # Save the history:
    with open('./models/seg_unet_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return model, history


def evaluate_model_seg():
    model = unet_mobilenetv2(input_shape=EXPECT_IMAGE_SHAPE, num_classes=1)
    model.load_weights('./models/seg_unet.h5')


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
        plt.imshow(images[i])

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(masks[i])

        plt.subplot(1, 3, 3)
        plt.title("Predicted mask")
        plt.imshow(y_pred[i])

        plt.show()


# prepare_mask_table()
# train_model_seg()

evaluate_model_seg()