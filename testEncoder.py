import tensorflow as tf
import numpy as np
import PIL.Image as Image
import random
import os
import time
import cv2
from tensorflow.python.ops.rnn_cell import LSTMCell
class LSTMAutoEncoder(object):
    def __init__(self, _X,
    LSTM_LAYERS = 2,
    withInputFlag= False,
    BATCH_SIZE = 10,
    NUM_CLASSES = 101,
    EPOCH_NUM = 10,
    VALIDATION_PRO = 0.2,
    n_input = 112*112*3,
    n_steps = 16,
    n_hidden = 64, #2048
    learning_rate = 0.0025,
    lambda_loss_amount = 0.0015):
        # np_mean = np.load('crop_mean.npy').reshape([n_steps, 112, 112, 3])
        self.n_steps = n_steps
        self.BATCH_SIZE = BATCH_SIZE
        self.n_input = n_input
        self.n_hidden = n_hidden
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
        self.oriX = _X
        self.followY = tf.transpose(tf.stack([_X[:,i+1, :] for i in range(self.n_steps-1)]),[1,0,2])
        self._X = _X
        self.encode()
        self.decode()
        self.prediction()
    # input: batch * n_step * n_input
    def encode(self):
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
        decode_inputs = tf.zeros([self.BATCH_SIZE, self.n_hidden], dtype = tf.float32)
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
        self.train = tf.train.AdamOptimizer().minimize(self.loss)
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
        self.predicts = self.predicts[:,1:,:]
        self.predLoss = tf.reduce_mean(tf.square(self.followY - self.predicts))
        self.predOpt = tf.train.AdamOptimizer().minimize(self.predLoss)

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)
# Constants
batch_num = 128
hidden_num = 12
step_num = 8
elem_num = 1
iteration = 1000
restore = True
_X = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
ae = LSTMAutoEncoder(_X, BATCH_SIZE = batch_num, n_hidden = hidden_num,\
                     n_input = elem_num, n_steps=step_num)
saver = tf.train.Saver()
with tf.Session() as sess:
    if(restore):
        saver.restore(sess, "models/testEncoderModel.ckpt")
        print("weights: %s" %ae.hiddenWeights.eval())
    else:
        sess.run(tf.global_variables_initializer())

        for i in range(iteration):
            r = np.random.randint(20, size=batch_num).reshape([batch_num, 1, 1])
            r = np.tile(r, (1, step_num, elem_num))
            d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num, elem_num])
            d = np.tile(d, (batch_num, 1, 1))
            random_sequences =  r+d

            (loss_val, _,) = sess.run([ae.loss, ae.train], {_X: random_sequences})
            (pred_loss, _,) = sess.run([ ae.predLoss, ae.predOpt], {_X: random_sequences})
            print('iter %d:' % (i + 1), loss_val, pred_loss)
            if(i % 10 == 0):
                save_path = saver.save(sess, "models/testEncoderModel.ckpt")
                print("Model saved in path: %s" % save_path)
    #test
    r = np.random.randint(20, size=batch_num).reshape([batch_num, 1, 1])
    r = np.tile(r, (1, step_num, elem_num))
    d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num, elem_num])
    d = np.tile(d, (batch_num, 1, 1))
    (input_, output_, pred_) = sess.run([ae.oriX, ae.outputs, ae.predicts], {_X:  r+d})
    print('train result :')
    print('input :', input_[0, :, :].flatten())
    print('output :', output_[0, :, :].flatten())
    print('predict: ', pred_[0].flatten())
