
from efficientnet.tfkeras import EfficientNetB0
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from keras_applications.inception_v3 import InceptionV3
import keras


from common import IMAGE_SHAPE

def binary_cnn_efficientnet():
    # Load the pre-trained EfficientNet model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE)

    # Create a custom classification head for the task
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Binary classification
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x) 

    # Construct the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model



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

def multilabel_inception(num_classes):    
    base_model = InceptionV3(
        include_top=False, 
        weights="imagenet", 
        input_shape=IMAGE_SHAPE,
        backend = keras.backend, 
        layers = keras.layers,
        models = keras.models, 
        utils = keras.utils)
        
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    out = keras.layers.Dense(num_classes, activation="sigmoid", name='dense_output')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=out)

    return model