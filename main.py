import tensorflow as tf
import numpy as np
import PIL.Image as Image
import random
import os
import time
import cv2

TRAIN_LIST_PATH = 'train.list'
TEST_LIST_PATH = 'test.list'
PERCEPTON_TRAIN_PATH = 'bin_data_32/train/'
PERCEPTON_TEST_PATH = 'bin_data_32/test/'
TRAIN_CHECK_POINT_PATH = 'classModel/train.ckpt'
ENCODER_CHECK_POINT_PATH = 'model/train.ckpt'

BATCH_SIZE = 8
n_hidden = 2048
NUM_CLASSES = 101
CLIP_LENGTH = 2
n_input = 112*112*3
percepton_input = 4096
CLIP_LENGTH = 16
EPOCH_NUM = 400
np_mean = np.load('crop_mean.npy').reshape([CLIP_LENGTH, 112, 112, 3])

class LSTMAutoEncoder(object):
    def __init__(self, _X,_labels,
    LSTM_LAYERS = 2,
    withInputFlag= False,
    BATCH_SIZE = 10,
    NUM_CLASSES = 101,
    n_input = 112*112*3,
    n_steps = 16,
    n_hidden = 64, #2048
    learning_rate = 0.0025):
        self.n_steps = n_steps
        self.BATCH_SIZE = BATCH_SIZE
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.lambda_loss_amount = 0.0015
        self.withInputFlag = withInputFlag
        self.encode_cell_unit = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)
        self.decode_cell_unit = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)
        self.pred_cell_unit = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)

        self.encode_cells = tf.contrib.rnn.MultiRNNCell([self.encode_cell_unit]*LSTM_LAYERS, state_is_tuple = True)
        self.decode_cells = tf.contrib.rnn.MultiRNNCell([self.decode_cell_unit]*LSTM_LAYERS, state_is_tuple = True)
        self.pred_cells = tf.contrib.rnn.MultiRNNCell([self.pred_cell_unit]*LSTM_LAYERS, state_is_tuple = True)

        self.hiddenWeights = tf.Variable(tf.random_normal([n_input, n_hidden]))
        self.outWeights = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], dtype=tf.float32))
        self.hiddenBiases = tf.Variable(tf.random_normal([n_hidden]))
        self.outBiases = tf.Variable(tf.constant(0.1, shape=[n_hidden], dtype=tf.float32))
        self.predWeights =  tf.Variable(tf.truncated_normal([n_hidden, n_hidden], dtype=tf.float32))
        self.predBiases = tf.Variable(tf.constant(0.1, shape=[n_hidden], dtype=tf.float32))
        self.classHiddenWeight = tf.Variable(tf.random_normal([n_input, n_hidden]))
        self.classOutWeight = tf.Variable(tf.random_normal([n_hidden, NUM_CLASSES], mean=1.0))
        self.classHiddenBiases = tf.Variable(tf.random_normal([n_hidden]))
        self.classOutBiases = tf.Variable(tf.random_normal([NUM_CLASSES]))
        self.oriX = _X[:,:,:percepton_input]
        self.followY = _X[:,:,percepton_input:]
        # self.followY = tf.transpose(tf.stack([_X[:,i+1, :] for i in range(self.n_steps-1)]),[1,0,2])
        self._X = _X[:,:,:percepton_input]
        self.batch_labels = _labels
        self.encode()
        self.decode()
        self.prediction()
        self.classification()


    def classification(self):
        self._X = self.oriX
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        for dataX in [self._X, self.followY]:
            dataX = tf.transpose(dataX, [1,0,2])
            dataX = tf.reshape(dataX, [-1, self.n_input])

            # Linear activation
            dataX = tf.nn.relu(tf.matmul(dataX, self.classHiddenWeight) + self.classHiddenBiases)
            # Split to n_steps' (batch * n_hidden), axis =0
            dataX = tf.split(dataX, self.n_steps)
            # self._X  = [tf.squeeze(t, [1]) for t in tf.split(self._X , self.n_steps, 1)]
            # print(len(self._X))
            # print(self._X[0].shape)
            with tf.variable_scope('classifier'):
                classOutputs, classStates = tf.contrib.rnn.static_rnn(self.encode_cells, dataX, dtype=tf.float32)
                lstm_last_output = classOutputs[-1]
                pred = tf.matmul(lstm_last_output, self.classOutWeight) + self.classOutBiases
                l2 = self.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
                # Softmax loss
                self.classCost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_labels, logits=pred)) + l2

                self.classOptimizer = optimizer.minimize(self.classCost)
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.batch_labels,1))
                self.classAccuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # input: batch * n_step * n_input
    def encode(self):
        self._X = self.oriX
        # change to (n_steps, batch_size, n_input)
        self._X = tf.transpose(self._X, [1,0,2])
        self._X = tf.reshape(self._X, [-1, self.n_input])

        # Linear activation
        self._X = tf.nn.relu(tf.matmul(self._X, self.hiddenWeights) + self.hiddenBiases)
        # Split to n_steps' (batch * n_hidden), axis =0
        self._X = tf.split(self._X, self.n_steps)
        # self._X  = [tf.squeeze(t, [1]) for t in tf.split(self._X , self.n_steps, 1)]
        # print(len(self._X))
        # print(self._X[0].shape)
        with tf.variable_scope('encoder'):
            self.aftCodes, self.encode_states = tf.contrib.rnn.static_rnn(self.encode_cells, self._X, dtype=tf.float32)
        # print(self.aftCodes[0].shape)
        # exit()
    def decode_with_input(self, vs):
        decode_states = self.encode_states
        decode_inputs = tf.zeros([self.BATCH_SIZE, self.n_hidden],dtype = tf.float32)
        dec_outs = []
        for step in range(self.n_steps):
            if(step>0):
                vs.reuse_variables()
            (decode_inputs, decode_states) = self.decode_cells(decode_inputs, decode_states)

            decode_inputs = tf.matmul(decode_inputs , self.outWeights) + self.outBiases
            # many to one
            dec_outs.append(tf.expand_dims(decode_inputs[:,-1],1))

        self.outputs = tf.transpose(tf.stack(dec_outs), [1, 0, 2])

    def decode_without_input(self):
        decode_inputs = [tf.zeros([self.BATCH_SIZE, self.n_hidden],dtype = tf.float32) for _ in range(self.n_steps)]
        (decode_outputs, decode_states) = tf.contrib.rnn.static_rnn(self.decode_cells,decode_inputs,\
                                            initial_state = self.encode_states,dtype=tf.float32)
        final_outputs = []
        for i, output in enumerate(decode_outputs):
            output= tf.matmul(output , self.outWeights) + self.outBiases
            output = tf.expand_dims(output[:,-1], 1)

            final_outputs.append(output)
        self.outputs = tf.transpose(tf.stack(final_outputs), [1, 0, 2])

        # dec_weights = tf.tile(tf.expand_dims(self.outWeights, 0), [self.BATCH_SIZE, 1, 1])
        # self.outputs = tf.matmul(dec_outs , dec_weights) + self.outBiases
    def decode(self):
        with tf.variable_scope('decoder') as vs:
            if(self.withInputFlag):
                self.decode_with_input(vs)
            else:
                self.decode_without_input()
        self.loss = tf.reduce_mean(tf.square(self.oriX - self.outputs))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    def prediction(self):
        decode_inputs = [tf.zeros([self.BATCH_SIZE, self.n_hidden],dtype = tf.float32) for _ in range(self.n_steps)]
        (decode_outputs, decode_states) = tf.contrib.rnn.static_rnn(self.pred_cells,decode_inputs,\
                                            initial_state = self.encode_states,dtype=tf.float32)
        final_outputs = []
        for i, output in enumerate(decode_outputs):
            output= tf.matmul(output , self.predWeights) + self.predBiases
            output = tf.expand_dims(output[:,-1], 1)

            final_outputs.append(output)
        self.predicts = tf.transpose(tf.stack(final_outputs), [1, 0, 2])
        self.predLoss = tf.reduce_mean(tf.square(self.followY - self.predicts))
        self.predOpt = tf.train.AdamOptimizer().minimize(self.predLoss)

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

def get_batches_perceptons(path, num_classes, batch_index, video_indices, batch_size=BATCH_SIZE):
    lines = os.listdir(path)
    clips = []
    labels = []
    for i in video_indices[batch_index: batch_index + batch_size]:
        label = lines[i].split('.')[0].split('_')[1]
        i_clip = np.fromfile(path + lines[i]).astype(np.float32)
        labels.append(int(label))
        clips.append(i_clip)
    clips = np.expand_dims(np.array(clips), axis=1)
    labels = np.array(labels)

    oh_labels = np.zeros([len(labels), num_classes]).astype(np.int64)
    for i in range(len(labels)):
        oh_labels[i, labels[i]] = 1
    batch_index = batch_index + batch_size
    batch_data = {'clips': clips, 'labels': oh_labels}
    return batch_data, batch_index

def run_AutoEncoderOnce(sess, trainValIndices, batch_index = 0):
    for i in range(len(trainValIndices[0]) // BATCH_SIZE):
        batch_data, batch_index = get_batches_perceptons(PERCEPTON_TRAIN_PATH, NUM_CLASSES, batch_index, trainValIndices[0])
        loss_out, accuracy_out = sess.run(
        [ae.loss, ae.train],
            feed_dict={percepton_clips:batch_data['clips']}
        )
        pred_loss, _ = sess.run(
        [ae.predLoss, ae.predOpt],
        feed_dict={percepton_clips:batch_data['clips']}
        )
    ae.classHiddenWeight = ae.hiddenWeights #tf.reduce_mean([ae.hiddenWeights, ae.predWeights],0)
    ae.classHiddenBiases = ae.hiddenBiases  #tf.reduce_mean([ae.hiddenBiases, ae.predBiases],0)
    print("en--de--pred")

batch_clips =  tf.placeholder(tf.float32, shape=(BATCH_SIZE, CLIP_LENGTH, n_input))
percepton_clips = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1, percepton_input*2))
batch_labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])

train_video_indices, validation_video_indices = get_video_indices(TRAIN_LIST_PATH)
train_losses = []
pred_losses = []
ae = LSTMAutoEncoder(percepton_clips,batch_labels, BATCH_SIZE = BATCH_SIZE, n_hidden = n_hidden,\
         n_input = percepton_input, n_steps=1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
trainValIndices = [train_video_indices, validation_video_indices]
trainValiName = ["training", "validation"]
saver = tf.train.Saver()
restore = True
with tf.Session(config=config) as sess:
    # if(restore):
    #     # saver.restore(sess, ENCODER_CHECK_POINT_PATH)
    # else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if(restore):
        run_AutoEncoderOnce(sess, trainValIndices)
    # ae.classification()
    for epoch in range(EPOCH_NUM):
        loss_epoch = 0
        acc_epoch = 0
        # pred_loss_epoch=0
        batch_index = 0
        for i in range(len(trainValIndices[0]) // BATCH_SIZE):
            # batch_data, batch_index = get_batches(TRAIN_LIST_PATH, NUM_CLASSES, batch_index,
            #                  trainValIndices[0], BATCH_SIZE,flatten=True)
            batch_data, batch_index = get_batches_perceptons(PERCEPTON_TRAIN_PATH, NUM_CLASSES, batch_index, trainValIndices[0])
            # loss_out, accuracy_out = sess.run(
            # [ae.loss, ae.train],
            #     feed_dict={percepton_clips:batch_data['clips']}
            # )
            # pred_loss, _ = sess.run(
            # [ae.predLoss, ae.predOpt], {percepton_clips:batch_data['clips']}
            # )
            _, loss_out, accuracy_out = sess.run(
            [ae.classOptimizer,ae.classCost, ae.classAccuracy],
                feed_dict={percepton_clips:batch_data['clips'],
                           batch_labels:batch_data['labels']}
            )
            loss_epoch += loss_out
            acc_epoch += accuracy_out
            # pred_loss_epoch+=pred_loss
            if i % 10 == 0:
                print('Epoch %d, Batch %d: Loss is %.5f Accuracy is : %.5f'%(epoch+1, i, loss_out, accuracy_out))

            # train_losses.append(loss_out)

            # pred_losses.append(pred_loss)

        print('Epoch %d: Average %s loss is: %.5f accuracy: %0.5f'%(epoch+1, trainValiName[0], loss_epoch / (len(trainValIndices[0]) // BATCH_SIZE), acc_epoch / (len(trainValIndices[0]) // BATCH_SIZE)))
        saver.save(sess, TRAIN_CHECK_POINT_PATH, global_step=epoch)
