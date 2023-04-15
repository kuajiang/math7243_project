# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:20:47 2023

@author: 22615
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D,BatchNormalization, Activation, Conv2DTranspose,concatenate

def pic_data(filepath='./'):
    Train = []
    Label = []
    
    paths = [
        './renders/epidural/brain_window/',
        # './renders/intraparenchymal/brain_window/',
        # './renders/intraparenchymal/brain_window/',
        # './renders/subarachnoid/brain_window/',
        # './renders/subdural/brain_window/'
    ]
    names = [
        './seg-label/epidural/',
        # './seg-label/intraparenchymal/',
        # './seg-label/intraventricular/',
        # './seg-label/subarachnoid/',
        # './seg-label/subdural/'
    ]
    for _ in range(len(paths)):
        for name in os.listdir(names[_]):
            file_path = paths[_]+name[6:-4]+'.jpg'
            if not os.path.exists(file_path):
                continue
            train_img = Image.open(paths[_]+name[6:-4]+'.jpg').resize((512,512))
            Train.append(np.array(train_img)/255)
            label_img = plt.imread(names[_]+name)
            Label.append(label_img)
    
    Train = np.array(Train)
    Label = np.expand_dims(np.array(Label), axis=3)
    np.random.seed(100)
    np.random.shuffle(Train)
    np.random.seed(100)
    np.random.shuffle(Label)
    print('Data already loaded')
    test_size = round(len(Train)*0.95)
    return [Train[:test_size,:,:],Label[:test_size,:,:],Train[test_size:,:,:],Label[test_size:,:,:]]
    
class U_net():
    def __init__(self):
        self.data = []
        self.dropout = 0.1
        self.input_size = (512,512,3)
        self.batch_size = 16
        self.u_net = self.buildNet()
        self.history = None
        
    def buildNet(self,filters=16):
        def block(tensor,filters):
            conv1 = Conv2D(filters,kernel_size=3,padding='same')(tensor)
            bn1 = BatchNormalization()(conv1)
            act1 = Activation('relu')(bn1)
            conv2 = Conv2D(filters,kernel_size=3,padding='same')(act1)
            bn2 = BatchNormalization()(conv2)
            act2 = Activation('relu')(bn2)
            return act2
            
        tf.keras.backend.clear_session()
        # construct starting from input layer
        inputs = Input(self.input_size)
        
        block1 = block(inputs,filters)
        pool1 =  MaxPooling2D(pool_size=(2,2))(block1)
        drop1 = Dropout(self.dropout*0.5)(pool1)
        
        block2 = block(drop1,filters*2)
        pool2 = MaxPooling2D(pool_size=(2,2))(block2)
        drop2 = Dropout(self.dropout)(pool2)
        
        block3 = block(drop2,filters*4)
        pool3 = MaxPooling2D(pool_size=(2,2))(block3)
        drop3 = Dropout(self.dropout)(pool3)
        
        block4 = block(drop3,filters*8)
        pool4 = MaxPooling2D(pool_size=(2,2))(block4)
        drop4 = Dropout(self.dropout)(pool4)        
        
        block5 = block(drop4,filters*16)
        pool5 = MaxPooling2D(pool_size=(2,2))(block5)
        drop5 = Dropout(self.dropout)(pool5)
        
        block6 = block(drop5,filters*32)
            
        # upsample staring from block7        
        up7 = Conv2DTranspose(filters*16,kernel_size=3,strides=(2,2),padding='same')(block6)
        up7 = concatenate([up7,block5])
        drop7 = Dropout(self.dropout)(up7)
        block7 = block(drop7,filters*16)
        
        up8 = Conv2DTranspose(filters*8,kernel_size=3,strides=(2,2),padding='same')(block7)
        up8 = concatenate([up8,block4])
        drop8 = Dropout(self.dropout)(up8)
        block8 = block(drop8,filters*8)       
        
        up9 = Conv2DTranspose(filters*4,kernel_size=3,strides=(2,2),padding='same')(block8)
        up9 = concatenate([up9,block3])
        drop9 = Dropout(self.dropout)(up9)
        block9 = block(drop9,filters*4)          
        
        up10 = Conv2DTranspose(filters*2,kernel_size=3,strides=(2,2),padding='same')(block9)
        up10 = concatenate([up10,block2])
        drop10 = Dropout(self.dropout)(up10)
        block10 = block(drop10,filters*2)
        
        up11 = Conv2DTranspose(filters,kernel_size=3,strides=(2,2),padding='same')(block10)
        up11 = concatenate([up11,block1])
        drop11 = Dropout(self.dropout)(up11)
        block11 = block(drop11,filters)
        
        outputs = Conv2D(filters=1,kernel_size=1,activation='sigmoid')(block11)    
        
        print('U-net is constructed')
        return tf.keras.models.Model(inputs,outputs)

    def metric_fun(self,y_true,y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        return fz / fm

    def train(self,epochs=100):
        ## lord trained weights of the model
        #checkpoints_path = './weights/'+os.listdir('./weights')[-1]

        # self.u_net.load_weights('./weights/my_Unet.h5')
        print("Weights are loaded")
        
        ## set training checkpoint
        train,label,test,tlabel = self.data
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/my_Unet.h5', save_best_only=True, save_freq='epoch')
        #checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/my_Unet_epoch{epoch:02d}.h5', save_best_only=True, save_freq='epoch')
        early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.0001,patience=50)
        ## train the model
        #self.u_net.compile(loss='mse',optimizer=Adam(0.01,0.5),metrics=[self.metric_fun])
        self.u_net.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(0.0005,0.5),metrics=[self.metric_fun])
        self.history = self.u_net.fit(train, label, batch_size=self.batch_size, epochs=epochs, 
                                      callbacks=[checkpoint,early_stop], validation_split=0.1, shuffle=True)
        
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        metric = self.history.history['metric_fun']
        val_metric = self.history.history['val_metric_fun']
        x = np.linspace(0, len(loss), len(loss))
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)      
        plt.plot(x, loss, x, val_loss)
        plt.title("Loss")
        plt.legend(['loss', 'val_loss'])
        plt.xlabel("Epochs")
        plt.ylabel("loss")

        plt.subplot(1,2,2) 
        plt.plot(x, metric, x, val_metric)
        plt.title("metric")
        plt.legend(['metric', 'val_metric'])
        plt.xlabel("Epochs")
        plt.ylabel("Dice")
        plt.show()
        
        return self.history
        
    def test(self):
        self.u_net.load_weights('./weights/my_Unet.h5')
        self.u_net.predict(self.data)
        
        return None
        

net = U_net()
net.u_net.summary()
# net.data = pic_data()
# net.train()
    #tf.keras.utils.plot_model(net.u_net,"my_Unet",show_shapes=True)

  