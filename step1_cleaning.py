import os
import time

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from common import load_images, split_df
from common import ALL_HEMORRHAGE_TYPES, ALL_IMAGE_DIRS


def check_file_exist():
    """Check if image files exist

    There are arround 750,000 records in the hemorrhage-labels.csv
    while only arround 120,000 files in the ./renders folder.
    Check if image files exist for records in hemorrhage-labels.csv
    For those exist, save to ./labels/hemorrhage-labels-exist.csv
    """
    table = pd.read_csv('./labels/hemorrhage-labels.csv')
    print("read csv file hemorrhage-labels.csv, total rows: ", len(table))

    # read all image file names to a set
    file_names = set()
    for column in ALL_IMAGE_DIRS:
        path = './renders/%s/brain_window' % column
        before_len = len(file_names)
        for file in os.listdir(path):
            file_names.add(file)
        print("count of ", column, len(file_names)-before_len)
    
    print("read all image file names, total: ", len(file_names))

    # check Image in file_names
    table["file_exists"] = ["yes" if x+".jpg" in file_names else "no" for x in table["Image"]]

    # filter by file_exists
    table = table[table['file_exists'] == 'yes']

    # save to a new csv file
    table.to_csv('./labels/hemorrhage-labels-exist.csv', index=False)
    print("save to csv file hemorrhage-labels-exist.csv, total rows: ", len(table))
    return table



def read_classify_lable():
    """Read lable data and set hemorrhage_type
    
    Read lable data from ./labels/hemorrhage-labels-exist.csv
    For convenience, add new columns 'hemorrhage_type' to the table
    The value of 'hemorrhage_type' is the column name where the value of 1, else set as 'normal'
    If multipile columns are 1, set 'hemorrhage_type' as 'multi'
    """

    table = pd.read_csv('./labels/hemorrhage-labels-exist.csv')
    print("read csv file hemorrhage-labels-exist.csv, total rows: ", len(table))

    # set normal as 1 where all other columns are 0
    table['normal'] = 1
    for column in ALL_HEMORRHAGE_TYPES:
        table.loc[table[column] == 1, 'normal'] = 0


    all_hemorrhage_types = ALL_HEMORRHAGE_TYPES.copy()
    all_hemorrhage_types.append('normal')
    print("add column 'normal' to the table")

    # set hemorrhage_type as the column name where the value of 1, else set as 'none'
    table['hemorrhage_type'] = table[all_hemorrhage_types].idxmax(axis=1)
    table['count'] = table[all_hemorrhage_types].sum(axis=1)
    idx_multi = table['count'] > 1
    table.loc[idx_multi, 'hemorrhage_type'] = 'multi'
    table = table.drop(columns=['count'])
    print("add column 'hemorrhage_type' to the table")

    # move 'exist' to the last column
    col = table.pop('file_exists')
    table.insert(len(table.columns), 'file_exists', col)

    # save csv file
    table.to_csv('./labels/hemorrhage-labels-exist.csv', index=False)

    return table

def load_cleaning_filenames():
    """Load image names for cleaning
    
    this image files are already labeled by myself, saved in ./cleaning/type%d folder
    type values a 0 to 4, with 2 the best quality, 0 and 4 worst
    """
    # read all file names to a dict
    file_name_type = dict()

    for i in range(0, 5):
        dir_path = './cleaning/type%d' % i
        for file in os.listdir(dir_path):
            file_name_type[file] = i
    print("total file names: ", len(file_name_type))
    return file_name_type


def save_cleaning_csv():
    """Save cleaning label filenames to hemorrhage-labels-cleaning.csv
    """

    # read csv file
    table = pd.read_csv('./labels/hemorrhage-labels-exist.csv')

    # load cleaning filenames
    file_names = load_cleaning_filenames()

    # set filename for cleaning model training
    table['filename'] = table['Image'].apply(lambda x: x + '.jpg')
    
    # filter by filename
    table = table[table['filename'].isin(file_names)]

    # set shape_type
    table['shape_type'] = table['filename'].apply(lambda x: file_names[x])


    table.to_csv('./labels/hemorrhage-labels-cleaning.csv', index=False)
    print("save to csv file hemorrhage-labels-cleaning.csv, total rows: ", len(table))

    print("stat iamge files prepared for cleaning model training")
    count_table = pd.crosstab(table['hemorrhage_type'], table['shape_type'], margins=True, margins_name='Total')
    print(count_table)

def build_VGG16(n_classes):
    # Load pre-trained VGG16 model without the top classification layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model (pre-trained VGG16)
    for layer in base_model.layers:
        layer.trainable = False

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_callback]
    )

    return history

def train_cleaning_model():
    # read csv file
    table = pd.read_csv('./labels/hemorrhage-labels-cleaning.csv')
    # table = table.sample(n=100, random_state=42)

    # split data to train, validation and test
    train_table, validate_table, test_table = split_df(table)

    # get Image from train_table
    image_file_paths = ['./renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image']) for _, row in train_table.iterrows()]
    # load images
    X_train = load_images(image_file_paths)
    # convert shape_type column to one-hot labels
    y_train = to_categorical(train_table['shape_type'].values, num_classes=5)
    print("train data shape: ", X_train.shape, y_train.shape)

    image_file_paths = ['./renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image']) for _, row in validate_table.iterrows()]
    X_val = load_images(image_file_paths)
    y_val = to_categorical(validate_table['shape_type'].values, num_classes=5)
    print("validation data shape: ", X_val.shape, y_val.shape)

    image_file_paths = ['./renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image']) for _, row in test_table.iterrows()]
    X_test = load_images(image_file_paths)
    y_test = to_categorical(test_table['shape_type'].values, num_classes=5)
    print("test data shape: ", X_test.shape, y_test.shape)

    # build model and train
    model = build_VGG16(5)
    history = train_model(model, X_train, y_train, X_val, y_val, 20, 32)

    # save model, add timestamp to the file name
    model_file_name = './models/cleaning-weight-%s.h5' % int(time.time())
    model.save(model_file_name)
    # save history
    history_file_name = './models/cleaning-history-%s.pkl' % int(time.time())
    pickle.dump(history.history, open(history_file_name, 'wb'))

    # Evaluate the model on the test data using `evaluate`
    print('Evaluate on test data')
    results = model.evaluate(X_test, y_test, batch_size=32)
    print('test loss, test acc:', results)

def test_model():
    # read csv file
    table = pd.read_csv('./labels/hemorrhage-labels-cleaning.csv')

    # split data to train, validation and test
    _, _, test_table = split_df(table)

    # load images
    image_file_paths = ['./renders/%s/brain_window/%s.jpg' % (row['hemorrhage_type'], row['Image']) for _, row in test_table.iterrows()]
    X_test = load_images(image_file_paths)
    # convert shape_type column to one-hot labels
    y_test = test_table['shape_type'].values
    print("test data shape: ", X_test.shape, y_test.shape)

    model = build_VGG16(5)
    model.load_weights('./models/cleaning-1681785936-weight.h5')
    # Evaluate the model on the test data using `evaluate`
    print('Evaluate on test data')
    y_predict = model.predict(X_test)
    y_predict = np.argmax(y_predict, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)
    print(f'Accuracy: {accuracy:.2f}')

    # Calculate macro-averaged precision
    precision = precision_score(y_test, y_predict, average='macro')
    print(f'Macro-averaged Precision: {precision:.2f}')

    # Calculate macro-averaged recall
    recall = recall_score(y_test, y_predict, average='macro')
    print(f'Macro-averaged Recall: {recall:.2f}')

    # Calculate macro-averaged F1 score
    f1 = f1_score(y_test, y_predict, average='macro')
    print(f'Macro-averaged F1 Score: {f1:.2f}')



if __name__ == "__main__":
    # prepare data
    # check_file_exist()
    # read_classify_lable()
    # load_cleaning_filenames()
    # save_cleaning_csv()
    
    # train model
    # train_cleaning_model()
    
    # test model
    test_model()