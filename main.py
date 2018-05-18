import tensorflow as tf 
import numpy as np
import PIL.Image as Image
import random
import os
import time
import cv2

TRAIN_LIST_PATH = 'train.list'
TEST_LIST_PATH = 'test.list'
TRAIN_CHECK_POINT_PATH = 'save_check_point/train.ckpt'
LSTM_LAYERS = 2
BATCH_SIZE = 10
NUM_CLASSES = 101
EPOCH_NUM = 10
VALIDATION_PRO = 0.2
n_input = 112*112*3
CLIP_LENGTH = 16
n_hidden = 64 #2048
learning_rate = 0.0025
lambda_loss_amount = 0.0015
np_mean = np.load('crop_mean.npy').reshape([CLIP_LENGTH, 112, 112, 3])

def get_video_indices(filename):
    lines = open(filename, 'r')
    #Shuffle data
    lines = list(lines)
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
    validation_video_indices = video_indices[:int(len(video_indices) * 0.2)]
    train_video_indices = video_indices[int(len(video_indices) * 0.2):]
    return train_video_indices, validation_video_indices
def get_test_num(filename):
    lines = open(filename, 'r')
    return len(list(lines))

def frame_process(clip, clip_length=CLIP_LENGTH, crop_size=112, channel_num=3):
    frames_num = len(clip)
    croped_frames = np.zeros([frames_num, crop_size, crop_size, channel_num]).astype(np.float32)


    #Crop every frame into shape[crop_size, crop_size, channel_num]
    for i in range(frames_num):
        img = Image.fromarray(clip[i].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x: crop_x + crop_size, crop_y : crop_y + crop_size, :]
        croped_frames[i, :, :, :] = img - np_mean[i]

    return croped_frames


def convert_images_to_clip(filename, clip_length=CLIP_LENGTH, crop_size=112, channel_num=3):
    clip = []
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        if len(filenames) < clip_length:
            for i in range(0, len(filenames)):
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
            for i in range(clip_length - len(filenames)):
                image_name = str(filename) + '/' + str(filenames[len(filenames) - 1])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
        else:
            s_index = random.randint(0, len(filenames) - clip_length)
            for i in range(s_index, s_index + clip_length):
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
    if len(clip) == 0:
       print(filename)
    clip = frame_process(clip, clip_length, crop_size, channel_num)
    return clip#shape[clip_length, crop_size, crop_size, channel_num]

def get_batches(filename, num_classes, batch_index, video_indices, batch_size=BATCH_SIZE, crop_size=112, channel_num=3, flatten=False):
    lines = open(filename, 'r')
    clips = []
    labels = []
    lines = list(lines)
    for i in video_indices[batch_index: batch_index + batch_size]:
        line = lines[i].strip('\n').split()
        dirname = line[0]
        label = line[1]
        i_clip = convert_images_to_clip(dirname, CLIP_LENGTH, crop_size, channel_num)
        if(flatten):
            clips.append(i_clip.reshape((CLIP_LENGTH,crop_size*crop_size*channel_num)))
#         print(i_clip.shape)
        labels.append(int(label))
    clips = np.array(clips).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    oh_labels = np.zeros([len(labels), num_classes]).astype(np.int64)
    for i in range(len(labels)):
        oh_labels[i, labels[i]] = 1
    batch_index = batch_index + batch_size
    batch_data = {'clips': clips, 'labels': oh_labels}
    return batch_data, batch_index
def get_test_num(filename):
    lines = open(filename, 'r')
    return len(list(lines))
'''
Inputs:
_X : input parameters, input shape: (batch_size, CLIP_LENGTH, n_input)
_weights and _biases: Linear activation

Function:
currently only stack 2 basic cells to construct RNN
'''
def LSTM_RNN(_X, _weights, _biases):
    # change to (CLIP_LENGTH, batch_size, n_input)
    _X = tf.transpose(_X, [1,0,2])
    _X = tf.reshape(_X, [-1, n_input])

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split to CLIP_LENGTH' (batch * n_hidden), axis =0
    _X = tf.split(_X, CLIP_LENGTH)

    lstmUnit = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_stack = []
    for i in range(LSTM_LAYERS):
        lstm_stack.append(lstmUnit)
    lstm_cells = tf.contrib.rnn.MultiRNNCell(lstm_stack, state_is_tuple = True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def training():
	train_losses = []
	train_accuracies = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	trainValIndices = [train_video_indices, validation_video_indices]
	trainValiName = ["training", "validation"]

	with tf.Session(config=config) as sess:
	    sess.run(tf.global_variables_initializer())
	    sess.run(tf.local_variables_initializer())
	    for epoch in range(EPOCH_NUM):
	        for split in range(2):
	            accuracy_epoch = 0
	            loss_epoch = 0
	            batch_index = 0
	            for i in range(len(trainValIndices[split]) // BATCH_SIZE):
	                batch_data, batch_index = get_batches(TRAIN_LIST_PATH, NUM_CLASSES, batch_index,
	                                 trainValIndices[split], BATCH_SIZE,flatten=True)

	                _, loss_out, accuracy_out = sess.run(
	                [optimizer, cost, accuracy],
	                    feed_dict={
	                        batch_clips:batch_data['clips'],
	                        batch_labels:batch_data['labels']
	                    }
	                )
	                loss_epoch += loss_out
	                accuracy_epoch += accuracy_out
	                if i % 10 == 0:
	                    print('Epoch %d, Batch %d: Loss is %.5f; Accuracy is %.5f'%(epoch+1, i, loss_out, accuracy_out))

	                train_losses.append(loss_out)
	                train_accuracies.append(accuracy_out)
	            print('Epoch %d: Average %s loss is: %.5f; Average accuracy is: %.5f'%(epoch+1, trainValiName[split], loss_epoch / (len(trainValIndices[split]) // BATCH_SIZE),
	                                                                                    accuracy_epoch / (len(trainValIndices[split]) // BATCH_SIZE)))
	        saver.save(sess, TRAIN_CHECK_POINT_PATH, global_step=epoch)

# def testing(restorer):
#     test_num = get_test_num(TEST_LIST_PATH)
#     test_video_indices = range(test_num)
# 	with tf.Session(config=config) as sess:
# 	    sess.run(tf.global_variables_initializer())
# 	    sess.run(tf.local_variables_initializer())
#         restorer.restore(sess, TRAIN_CHECK_POINT)
# 	    accuracy_epoch = 0
# 	    batch_index = 0
# 	    for i in range(test_num // BATCH_SIZE):
# 	        if i % 10 == 0:
# 	            print('Testing %d of %d'%(i + 1, test_num // BATCH_SIZE))
# 	        batch_data, batch_index = get_batches(TEST_LIST_PATH, NUM_CLASSES, batch_index,
# 	                 test_video_indices, BATCH_SIZE,flatten=True)
# 	        pred_out, accuracy_out = sess.run(
# 	            [pred, accuracy],
# 	            feed_dict={
# 	                batch_clips:batch_data['clips'],
# 	                batch_labels:batch_data['labels']
# 	            }
# 	        )
# 	        accuracy_epoch+=accuracy_out
# 	print('Test accuracy is %.5f' % (accuracy_epoch / (test_num // BATCH_SIZE)))


train_video_indices, validation_video_indices = get_video_indices(TRAIN_LIST_PATH)
# Graph input/output
batch_clips = tf.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, n_input])
batch_labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, NUM_CLASSES], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
}

pred = LSTM_RNN(batch_clips, weights, biases)

# Loss, optimizer and evaluation
# L2 loss prevents this overkill neural network to overfit the data
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

# Softmax loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels, logits=pred)) + l2 
 # Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(batch_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
training()
testing(saver)