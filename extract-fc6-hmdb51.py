# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import math
import numpy as np
from tools import placeholder_inputs
from tools import _variable_with_weight_decay
from tools import tower_loss
from tools import tower_acc
from tools import _variable_on_cpu
from tools import _variable_with_weight_decay
from tools import average_gradients
from tools import get_logits
import input_data_for_extract_feature as input_data
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS
model_name = "sports1m_finetuning_ucf101.model"
unit_of_clip=2

def run_test(listfile, data_dir):
    num_test_videos = len(list(open(listfile,'r')))
    print("Number of test videos={}".format(num_test_videos))

    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
    feature_placeholder, _ = placeholder_inputs(FLAGS.batch_size * gpu_num*unit_of_clip)

    fc6 = get_logits(images_placeholder, labels_placeholder,FLAGS,gpu_num)
    saver = tf.train.Saver()


    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)

#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    init = tf.global_variables_initializer()######
    sess.run(init)
    saver.restore(sess, model_name)

    next_batch_start = 0
    all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)


#     writer = tf.python_io.TFRecordWriter('fc6.tfrecord')
    file_index = 0
    fc6_batch_size=FLAGS.batch_size
    for step in xrange(all_steps):
#         try:
        np_arr_data, np_arr_label, next_batch_start, _, _ = input_data.read_clip_and_label(
                    test_list_file,FLAGS.batch_size,unit_of_clip,start_pos=next_batch_start)
        fc6_of_batch=[]
        print("np_arr_data",np.shape(np_arr_data))
        if(np_arr_data.shape[0] == 0):
            print("give up this")
            continue;
        batch_seq_arr = np_arr_data.reshape((fc6_batch_size * unit_of_clip, 16,112,112,3))
        print("batch_seq_arr",np.shape(batch_seq_arr))

        for batch_unit_index in range(0,unit_of_clip):
            temp_batch_data = batch_seq_arr[batch_unit_index*fc6_batch_size:batch_unit_index*fc6_batch_size + fc6_batch_size]
            batch_seq_fc6 = sess.run(fc6,feed_dict={images_placeholder:temp_batch_data})
            fc6_of_batch.append(batch_seq_fc6)
            #print("np.shape(batch_seq_fc6)",np.shape(batch_seq_fc6))
#         print("np.shape(fc6_of_batch)",np.shape(fc6_of_batch))
        fc6_of_batch = np.array(fc6_of_batch)
        fc6_feature_batch = fc6_of_batch.reshape((fc6_batch_size,unit_of_clip * 4096))
#         print("np.shape(fc6_feature_batch)",np.shape(fc6_feature_batch))
#         print("np_arr_label",np_arr_label)
        #print(fc6_feature_batch)

        for batch_index in range(fc6_batch_size):
            try:
                #image = io.imread(images[i]) # type(image) must be array!
                data = fc6_feature_batch[batch_index]
                label = np_arr_label[batch_index]
                file_index += 1
                filename = "%s/%08d_%02d.bin" % (data_dir, file_index, label)
                # print("data-->",data.shape)
                # print("label-->",label)
                # print("filename",filename)
                with open(filename, 'wb') as f:
                    f.write(data)
            except IOError as e:
                print('Skip it!\n')

    print("Done!")

def main(_):
    split = ['train', 'test']
    for item in split
        list_file = 'hmdb-list/' + item +'.list'
        data_dir = "./bin_data_hmdb51/' + item +'/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        run_test(list_file,data_dir)
        print(item + ' finish')

if __name__ == '__main__':
    tf.app.run()
