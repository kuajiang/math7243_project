import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_images
from common import RANDOM_SEED, ALL_IMAGE_DIRS, ALL_HEMORRHAGE_TYPES, IMAGE_SHAPE
from step1_cleaning import build_VGG16
from step2_classify import multilabel_mobilenet
from step3_segmentation import unet_mobilenetv2, read_mask_file

def select_test_images():
    # these images are not used in classify training
    df = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    classify_test_df = df.sample(frac=0.2, random_state=RANDOM_SEED)

    # these images are not used in segmentation training
    segmentation_test_df = df.sample(frac=0.1, random_state=RANDOM_SEED)
    df_mask = pd.read_csv('./labels/hemorrhage-labels-mask.csv')
    df_mask["Image"] = df_mask["image"]
    segmentation_test_df = df_mask.sample(frac=0.1, random_state=RANDOM_SEED)
    
    # common subset of classify and segmentation test images
    common_test_df = classify_test_df.merge(segmentation_test_df, on=['Image', 'hemorrhage_type', 'shape'])

    test_dfs = []

    for image_dir in ALL_IMAGE_DIRS:
        if image_dir == 'normal':
            continue
        # filter from common_test_df by hemorrhage type
        df_by_type = common_test_df[common_test_df['hemorrhage_type'] == image_dir]
        # select 5 images from each type
        df_sample = df_by_type.sample(n=5, random_state=RANDOM_SEED)
        test_dfs.append(df_sample)
    
    # select 10 normal images
    df_normal = classify_test_df[classify_test_df['hemorrhage_type'] == 'normal'].sample(n=10, random_state=RANDOM_SEED)
    test_dfs.append(df_normal)

    # combine all test images
    test_df = pd.concat(test_dfs)

    # print 5 image name every line
    image_names = [row['Image'] for index, row in test_df.iterrows()]
    for i in range(0, len(image_names), 5):
        print(image_names[i:i+5])

    # copy images to test_images directory
    for index, row in test_df.iterrows():
        image_file_path = "./renders/%s/brain_window/%s.jpg" % (row['hemorrhage_type'], row['Image'])
        print(image_file_path)
        
        target_file_path = "./test_images/%s.jpg" % row['Image']
        # copy image
        shutil.copyfile(image_file_path, target_file_path)

        if row['hemorrhage_type'] == 'normal':
            continue

        # copy mask
        mask_file_path = "./seg-label/%s/label_%s.png" % (row['hemorrhage_type'], row['Image'])
        mask_target_path = "./test_images/%s_mask.png" % row['Image']
        shutil.copyfile(mask_file_path, mask_target_path)



def test_image(image_name):
    image_path = "./test_images/%s.jpg" % image_name
    # check file exist
    if not os.path.isfile(image_path):
        print("Image %s not found" % image_path)
        sys.exit(1)
    print(image_path)
    
    # read image
    images = load_images([image_path])
    # test image valid
    model_cleaning = build_VGG16(len(ALL_HEMORRHAGE_TYPES))
    model_cleaning.load_weights('./models/cleaning-1681785936-weight.h5')
    # predict
    valid_predictions = model_cleaning.predict(images)
    # print predictions
    valid_value = np.argmax(valid_predictions, axis=1)
    if valid_value in [0, 4]:
        print("Data cleaning classification result : %d" % valid_value)
        print("Image %s is not valid for following classification and segmentation" % image_name)
        sys.exit(1)
    print("Image %s is valid for following classification and segmentation, value = %d" % (image_name, valid_value))

    # classify
    model_classify = multilabel_mobilenet(len(ALL_HEMORRHAGE_TYPES))
    model_classify.load_weights('./models/classify_weights_epoch_12.h5')
    # predict
    classify_predictions = model_classify.predict(images)
    threshold = 0.5
    predict_value = (classify_predictions >= threshold).astype(int)[0]

    # find from label by image name
    df = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    df = df[df['Image'] == image_name]
    real_value = df[ALL_HEMORRHAGE_TYPES].values[0]

    real_type = hemorrhage_types_to_str(real_value)
    
    print("Real Hemorrhage Type : %s" % real_type)
    print("Predict Hemorrhage Type : %s" % hemorrhage_types_to_str(predict_value))

    if real_type == 'normal':
        print("Image %s is normal, no need to do segmentation" % image_name)
        return

    # segmentation
    model_segmentation = unet_mobilenetv2(input_shape=IMAGE_SHAPE, num_classes=1)
    model_segmentation.load_weights('./models/seg_unet_1550_59_weight.h5')
    # predict
    segmentation_predictions = model_segmentation.predict(images*255.0)
    print(segmentation_predictions.shape)

    # read mask
    mask_file_path = "./test_images/%s_mask.png" % image_name
    # check mask file exist
    if not os.path.isfile(mask_file_path):
        print("Mask %s not found" % mask_file_path)
        sys.exit(1)
    masks = read_mask_file(mask_file_path)

    # Plot the results
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(images[0])

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(masks)

    plt.subplot(1, 3, 3)
    plt.title("Predicted mask")
    plt.imshow(segmentation_predictions[0])
    
    # print the whole array
    np.set_printoptions(threshold=sys.maxsize)

    plt.show()




def hemorrhage_types_to_str(hemorrhage_types):
    result =  ','.join([ALL_HEMORRHAGE_TYPES[i] for i in range(len(hemorrhage_types)) if hemorrhage_types[i] == 1])
    if not result:
        result = 'normal'
    return result


def select_random_test_image():
    # select image file from test_images directory with .jpg extension
    test_images = [f for f in os.listdir('./test_images') if f.endswith('.jpg')]
    # select random image
    image_name = np.random.choice(test_images)
    # remove .jpg extension
    image_name = image_name[:-4]
    return image_name

# main
if __name__ == "__main__":
    # select_test_images()

    # read image name from command line
    if len(sys.argv) < 2:
        image_name = select_random_test_image()
        print("image name not specified, using random test image %s" % image_name)
    else:
        image_name = sys.argv[1]

    test_image(image_name)