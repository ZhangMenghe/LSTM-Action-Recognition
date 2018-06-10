# -*- coding: utf-8 -*-
"""C3D model for Keras

# Reference:

- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)

Based on code from @albertomontesg
"""

# import skvideo.io
import os
from six.moves import xrange
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.utils.data_utils import get_file
from keras.layers.core import Dense, Dropout, Flatten
# from sports1M_utils import preprocess_input, decode_predictions
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
import input_data_for_extract_feature as input_data

WEIGHTS_PATH = 'sports1m/sports1M_weights_tf.h5'
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS
unit_of_clip=2

def C3D(weights='sports1M'):
    """Instantiates a C3D Kerasl model

    Keyword arguments:
    weights -- weights to load into model. (default is sports1M)

    Returns:
    A Keras model.

    """

    if weights not in {'sports1M', None}:
        raise ValueError('weights should be either be sports1M or None')

    if K.image_data_format() == 'channels_last':
        shape = (16,112,112,3)
    else:
        shape = (3,16,112,112)

    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))

    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))

    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))

    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))

    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0,1,1)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if weights == 'sports1M':
        #weights_path = get_file('sports1M_weights_tf.h5',
         #                       WEIGHTS_PATH,
          #                      cache_subdir='models',
           #                     md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')

        model.load_weights(WEIGHTS_PATH)

    return model
def run(model, listfile, data_dir):
    num_test_videos = len(list(open(listfile,'r')))
    print("Number of test videos={}".format(num_test_videos))

    next_batch_start = 0
    all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)

    file_index = 0
    fc6_batch_size=FLAGS.batch_size
    for step in xrange(all_steps):
        np_arr_data, np_arr_label, next_batch_start, _, _ = input_data.read_clip_and_label(
            listfile,FLAGS.batch_size,unit_of_clip,start_pos=next_batch_start)
        fc6_of_batch=[]
        print("np_arr_data",np.shape(np_arr_data))
        if(np_arr_data.shape[0] == 0):
            print("give up this")
            continue;
        batch_seq_arr = np_arr_data.reshape((fc6_batch_size * unit_of_clip, 16,112,112,3))
        print("batch_seq_arr",np.shape(batch_seq_arr))

        for batch_unit_index in range(0,unit_of_clip):
            temp_batch_data = batch_seq_arr[batch_unit_index*fc6_batch_size:batch_unit_index*fc6_batch_size + fc6_batch_size]
            batch_seq_fc6 = model.predict(temp_batch_data, batch_size = 2)
            fc6_of_batch.append(batch_seq_fc6)
        fc6_of_batch = np.array(fc6_of_batch)
        fc6_feature_batch = fc6_of_batch.reshape((fc6_batch_size,unit_of_clip * 4096))
        for batch_index in range(fc6_batch_size):
            try:
                data = fc6_feature_batch[batch_index]
                label = np_arr_label[batch_index]
                file_index += 1
                filename = "%s/%08d_%02d.bin" % (data_dir, file_index, label)
                with open(filename, 'wb') as f:
                    f.write(data)
            except IOError as e:
                print('Skip it!\n')
    print("Done!")
if __name__ == '__main__':
    model = C3D(weights='sports1M')
    fc6 = Model(inputs = model.input, outputs = model.get_layer('fc6').output)
    split = ['train', 'test']
    for item in split:
        list_file = 'list/' + item +'.list'
        data_dir = './bin_data_ucf101/' + item
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        run(fc6, list_file,data_dir)
        print(item + ' finish')
    
    # for layer in fc6.layers:
    #     weights = layer.get_weights()
    #     print(weights)

    # 
    # vid_path = 'homerun.mp4'
    # vid = skvideo.io.vread(vid_path)
    # vid = vid[40:56]
    # vid = preprocess_input(vid)
    #
    # preds = model.predict(vid)
    # print(decode_predictions(preds))
