import os
from sklearn.model_selection import KFold
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, Bidirectional, Dropout
from keras.layers import Reshape, Lambda, BatchNormalization,Concatenate
from keras import applications
from keras.layers.recurrent import LSTM,GRU
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras import layers,models
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from loader import TextImageGenerator, MAX_LEN, CHAR_DICT, SIZE, VizCallback, ctc_lambda_func
import numpy as np
import tensorflow as tf
from keras import backend as K
import argparse
K.tensorflow_backend._get_available_gpus()

# def model_cnn(inputs):
# 	#inputs = Input(name='the_inputs', shape=(1600,64,3), dtype='float32')
# 	x = layers.Conv2D(64, (3, 3),
#                   activation='relu',
#                   padding='same',
#                   name='block1_conv1')(inputs)
# 	x = layers.Conv2D(64, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block1_conv2')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# 	# Block 2
# 	x = layers.Conv2D(128, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block2_conv1')(x)
# 	x = layers.Conv2D(128, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block2_conv2')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# 	# Block 3
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block3_conv1')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block3_conv2')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block3_conv3')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# 	# Block 4
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block4_conv1')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block4_conv2')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block4_conv3')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 1), name='block4_pool')(x)

# 	# Block 5
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv1')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv2')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv3')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 1), name='block5_pool')(x)


# 	model = models.Model(inputs, x, name='cnn')
# 	return model


def model_cnn(inputs):
	#inputs = Input(name='the_inputs', shape=(120,32,3), dtype='float32')
	x1 = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(inputs)
	x1 = layers.Conv2D(64, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block1_conv2')(x1)
	x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

	# Block 2
	x1 = layers.Conv2D(128, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block2_conv1')(x1)
	x1 = layers.Conv2D(128, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block2_conv2')(x1)
	x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x1)
	skip_1 = x1

	# Block 3
	x1 = layers.Conv2D(256, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block3_conv1')(x1)
	x1 = layers.Conv2D(256, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block3_conv2')(x1)
	y_1 = Concatenate(axis=-1)([skip_1,x1])
	x1 = layers.MaxPooling2D((2,2), strides=(2, 2), name='block3_pool')(y_1)
	skip_2 = x1
	# Block 4
	x1 = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv1')(x1)
	x1 = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv2')(x1)
	x1 = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv3')(x1)
	y_2 =  Concatenate(axis=-1)([skip_2,x1])
	x1 = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')(y_2)

	# Block 5
	x1 = layers.Conv2D(512, kernel_size=(1, 2),strides=(1,2),
	                  activation='relu',
	                  padding='valid',
	                  name='block5_conv1')(x1)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv2')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv3')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 1), name='block5_pool')(x)
	x2 = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv11')(inputs)
	x2 = layers.Conv2D(64, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block1_conv21')(x2)
	x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x2)

	# Block 2
	x2 = layers.Conv2D(128, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block2_conv11')(x2)
	x2 = layers.Conv2D(128, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block2_conv21')(x2)
	x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(x2)

	# Block 3
	x2 = layers.Conv2D(256, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block3_conv11')(x2)
	x2 = layers.Conv2D(256, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block3_conv21')(x2)
	x2 = layers.MaxPooling2D((2,2), strides=(2, 2), name='block3_pool1')(x2)

	# Block 4
	x2 = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv11')(x2)
	x2 = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv21')(x2)
	x2 = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv31')(x2)
	x2 = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool1')(x2)

	# Block 5
	x2 = layers.Conv2D(512, kernel_size=(1, 2),strides=(1,2),
	                  activation='relu',
	                  padding='valid',
	                  name='block5_conv11')(x2)
	x =  Concatenate(axis=-1)([x1,x2])
	model = models.Model(inputs, x, name='cnn')
	return model



def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = model_cnn(inputs)
    #base_model.trainable = False
    inner = base_model(inputs)
    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)
    
    inner =  BatchNormalization()(inner)
    inner = Dropout(0.2)(inner) 
    #lstm = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.5, recurrent_dropout=0.25))(inner)
    
    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense3')(inner)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # for layer in base_model.layers:
    #     layer.trainable = True
    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)

def train( datapath, labelpath,  epochs, batch_size, lr, finetune):
    sess = tf.Session()
    K.set_session(sess)

    model, y_func = get_model((*SIZE, 3), training=True, finetune=finetune)
    ada = Adam(lr=lr,clipvalue=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    ## load data
    id =  np.arange(len(os.listdir(datapath)))
    train_idx, valid_idx = train_test_split(id,test_size=0.05, random_state=42)
    train_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 8, train_idx, True, MAX_LEN)
    #train_generator.build_data()
    valid_generator  = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 8, valid_idx, False, MAX_LEN)
    #valid_generator.build_data()

    ## callbacks
    weight_path = 'model/best_weight.h5'
    ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    vis = VizCallback(sess, y_func, valid_generator, len(valid_idx))
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='min')

    if finetune:
        print('load pretrain model')
        model.load_weights(weight_path)
    
    # for layer in model.layers:
    #     if layer.name == 'cnn':
    #         layer.trainable = False
    #         model.summary()
    model.fit_generator(generator=train_generator.next_batch(),
                    steps_per_epoch=int(len(train_idx) / batch_size),
                    epochs=epochs,
                    callbacks=[ckp, vis, earlystop],
                    validation_data=valid_generator.next_batch(),
                    validation_steps=int(len(valid_idx) / batch_size))
    
# def train(datapath, labelpath, epochs, batch_size, lr, finetune=False):
#     nsplits = 2

#     nfiles = np.arange(len(os.listdir(datapath)))

#     kfold = list(KFold(nsplits, random_state=2018).split(nfiles))
#     for idx in range(nsplits):
#         train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='images/', type=str)
    parser.add_argument("--label", default='labels_out/labels.json', type=str)

    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
   	

    train(args.train, args.label, args.epochs, args.batch_size, args.lr, args.finetune)
    
