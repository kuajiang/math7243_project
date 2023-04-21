import time

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from common import load_image_data_by_df, stat_shape
from common import IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES, RANDOM_SEED

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


def train_model(model, train_df, valid_df):
    batch_size = 64

    train_generator = ClassifyDataGenerator(train_df, batch_size, IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES)
    valid_generator = ClassifyDataGenerator(valid_df, batch_size, IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create an EarlyStopping callback that monitors the validation accuracy
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


    # Train the model using the custom data generators
    num_epochs = 20  # Set the number of epochs as needed
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

def multilabel_mobilenet(num_classes):
    # Load the pre-trained MobileNetV2 model without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE)

    # Add a global average pooling layer
    x = GlobalAveragePooling2D()(base_model.output)

    # Add a custom classification head with a sigmoid activation for multilabel classification
    output = Dense(num_classes, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    return model

def train_classify(data_cleaning=False):
    """Train a multilabel classification model to classify hemorrhage types
    
    Arguments:
        data_cleaning {bool} -- whether to clean the data, default is False
    """
    df = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    print("total records: ", len(df))
    stat_shape(df)

    # use all data
    df_after_select = df
    if data_cleaning:
        df_after_select = df[df['shape'].isin([1,2,3])]    
    # df_after_select = df.sample(100, random_state=RANDOM_SEED)

    # split into training and validation sets
    test_df = df_after_select.sample(frac=0.2, random_state=RANDOM_SEED)
    remain_df = df_after_select.drop(test_df.index)
    train_df = remain_df.sample(frac=0.8, random_state=RANDOM_SEED)
    valid_df = remain_df.drop(train_df.index)

    # build multilabel classification model
    model = multilabel_mobilenet(len(ALL_HEMORRHAGE_TYPES))

    # train model
    train_model(model, train_df, valid_df)



def test_model(shape_list=[2]):
    """Test the model with test data

    Arguments:
        shape_list {list} -- the shape list to test, default is [2]
    """
    df = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    print("total records: ", len(df))
    stat_shape(df)

    # select all
    df_after_select = df
#     df_after_select = df.sample(100, random_state=RANDOM_SEED)

    # same split as training, make sure the test data is not used in training
    test_df = df_after_select.sample(frac=0.2, random_state=RANDOM_SEED)    
    test_df = test_df[test_df['shape'].isin(shape_list)]
    print("test records for test: ", len(test_df))
    stat_shape(test_df)

    # build model
    model = multilabel_mobilenet(len(ALL_HEMORRHAGE_TYPES))
    model.load_weights('./models/classify_weights_epoch_12.h5')

    # predict
    test_generator = ClassifyDataGenerator(test_df, 64, IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES)
    y_pred_proba = model.predict(test_generator)

    # evaluate accuracy
    y_true = test_df[ALL_HEMORRHAGE_TYPES]

    # Set a threshold to convert probabilities to binary predictions
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred_proba, average='micro', multi_class='ovr')

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Calculate the accuracy for each label
    label_accuracies = []
    for label_idx in range(y_true.shape[1]):
        accuracy = accuracy_score(y_true.values[:, label_idx], y_pred[:, label_idx])
        label_accuracies.append(accuracy)

    # Print the accuracies for each label
    for label_idx, accuracy in enumerate(label_accuracies):
        print(f"Label {label_idx + 1} accuracy: {accuracy:.4f}")


def main():
    # train_classify(False)
    # train_classify(True)
    test_model()


if __name__ == '__main__':
    main()