

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


from common import load_image_data_by_df, stat_shape, select_by_shape
from common import IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES, RANDOM_SEED

from build_models import multilabel_mobilenet, multilabel_inception

from common import load_image_data_by_df, stat_shape, select_by_shape
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




def multi_classify_evaluate():
    df = pd.read_csv('./labels/hemorrhage-labels-shape.csv')
    print("total records: ", len(df))
    stat_shape(df)

    # select all
    df_after_select = df
#     df_after_select = df.sample(100, random_state=RANDOM_SEED)

    df = df[df['shape'].isin([1,2,3])]

    # Split the DataFrame into training and validation sets
    test_df = df_after_select.sample(frac=0.2, random_state=RANDOM_SEED)
    

    # build model
    label_fields = ALL_HEMORRHAGE_TYPES
    model = multilabel_mobilenet(len(label_fields))
    model.load_weights('./models/classify_multi_19_66.h5')
    model.load_weights('./models/classify_weights_epoch_12.h5')

    
    # predict
    test_generator = ClassifyDataGenerator(test_df, 64, IMAGE_SHAPE, ALL_HEMORRHAGE_TYPES)
    y_pred = model.predict(test_generator)

    # evaluate accuracy
    y_true = test_df[ALL_HEMORRHAGE_TYPES]
    
    return y_true, y_pred

    

y_true, y_pred_proba = multi_classify_evaluate()
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


# print(y_true)
# print(y_pred)

print(y_true.values[:, 1])
print(y_pred[:, 1])
print(accuracy_score(y_true.values[:, 1], y_pred[:, 1]))




# Calculate the accuracy for each label
label_accuracies = []
for label_idx in range(y_true.shape[1]):
    accuracy = accuracy_score(y_true.values[:, label_idx], y_pred[:, label_idx])
    label_accuracies.append(accuracy)

# Print the accuracies for each label
for label_idx, accuracy in enumerate(label_accuracies):
    print(f"Label {label_idx + 1} accuracy: {accuracy:.4f}")