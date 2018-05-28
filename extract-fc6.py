import os
import time
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

model_name = "sports1m_finetuning_ucf101.model"


BATCH_SIZE = 2
CLIP_LENGTH = 16
SEQ_NUM = 2
def run_test(listFileName, storeDir):
    num_test_videos = len(list(open(listFileName,'r')))
    print("Number of test videos={}".format(num_test_videos))

    images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE,num_frames_per_clip =CLIP_LENGTH)
    feature_placeholder, _ = placeholder_inputs(BATCH_SIZE * CLIP_LENGTH)

    fc6 = get_logits(images_placeholder, labels_placeholder, BATCH_SIZE)
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
    all_steps = int((num_test_videos - 1) / BATCH_SIZE + 1)


    file_index = 0
    for step in xrange(all_steps):
        np_arr_data, np_arr_label, next_batch_start, _, _ = input_data.read_clip_and_label(
                            listFileName,BATCH_SIZE,SEQ_NUM,start_pos=next_batch_start,num_frames_per_clip=CLIP_LENGTH)
        fc6_of_batch = []
        # print("new:")
        # print(np_arr_data.shape)
        # batch_size,seq_num,num_frames_per_clip,crop_size,crop_size,3
        for i in range(SEQ_NUM):
            batch_seq_fc6 = sess.run(fc6, feed_dict={images_placeholder:np_arr_data[:,i,:,:,:]})
            fc6_of_batch.append(batch_seq_fc6)
        fc6_of_batch = np.array(fc6_of_batch)
        fc6_feature_batch = fc6_of_batch.reshape((-1,SEQ_NUM,4096))
        for batch_index in range(min(BATCH_SIZE, np_arr_label.shape[0])):
            try:
                #image = io.imread(images[i]) # type(image) must be array!
                data = fc6_feature_batch[batch_index]
                data = data.astype(np.float64)
                label = np_arr_label[batch_index]
                file_index += 1
                filename = "%s/%08d_%02d.bin" % (storeDir, file_index,label)
                # print("data-->",data)
                # print("label-->",label)
                # print("filename",filename)
                # with open(filename, 'wb') as f:
                #     f.write(data[i,:])
                data.tofile(filename)
            except IOError as e:
                print('Skip it!\n')
def main():
    spliting = ["train","test"]
    # spliting = ["debug"]
    for item in spliting:
        data_dir = "bin_data_32/" + item
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        run_test(item+".list", data_dir)
        print(item+" finish!\n")
if __name__ == '__main__':
    main()
