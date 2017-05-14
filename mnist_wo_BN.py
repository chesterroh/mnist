#!/usr/bin/python3

from mnist import MnistData
import tensorflow as tf
import numpy as np

pickle_file = 'notMNIST_sanit.pickle'
mnist = MnistData(pickle_file,one_hot=True)

def inference(input,is_training):

    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1,shape=shape))

    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    def conv_2_block(input,i_ch,o_ch,name):
        with tf.name_scope(name):
            w_conv1 = weight_variable([3,3,i_ch,o_ch])
            b_conv1 = bias_variable([o_ch])
            h_conv1 = tf.nn.relu(conv2d(input,w_conv1)+b_conv1)
            w_conv2 = weight_variable([3,3,o_ch,o_ch])
            b_conv2 = bias_variable([o_ch])
            h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2)+b_conv2)
            h_pool = max_pool_2x2(h_conv2)
            return h_pool
    
    def conv_3_block(input,i_ch,o_ch,name):
        with tf.name_scope(name):
            w_conv1 = weight_variable([3,3,i_ch,o_ch])
            b_conv1 = bias_variable([o_ch])
            h_conv1 = tf.nn.relu(conv2d(input,w_conv1)+b_conv1)
            w_conv2 = weight_variable([3,3,o_ch,o_ch])
            b_conv2 = bias_variable([o_ch])
            h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2)+b_conv2)
            w_conv3 = weight_variable([3,3,o_ch,o_ch])
            b_conv3 = bias_variable([o_ch])
            h_conv3 = tf.nn.relu(conv2d(h_conv2,w_conv3)+b_conv3)
            h_pool = max_pool_2x2(h_conv3)
            return h_pool
    
    def fc_block(input,i_shape,o_shape,name,relu=True):
        with tf.name_scope(name):
            w_fc = weight_variable([i_shape,o_shape])
            b_fc = bias_variable([o_shape])
            if relu:
                return tf.nn.relu(tf.matmul(input,w_fc)+b_fc)
            else:
                return tf.matmul(input,w_fc)+b_fc
    # beginning of the graph

    net = input
    net = conv_2_block(net,1,32,"conv_block1")
    net = conv_2_block(net,32,64,"conv_block2")
    net = conv_3_block(net,64,128,"conv_block3")
    net = conv_3_block(net,128,256,"conv_block4")
    net = tf.reshape(net,[-1,2*2*256])
    net = fc_block(net,2*2*256,512,"fc_block1")
    net = fc_block(net,512,512,"fc_block2")
    net = fc_block(net,512,10,"final_fc_block",relu=False)
    return net

def cross_entropy(y,y_):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
        tf.summary.scalar('cross_entropy',cross_entropy)
        return cross_entropy

def train(net,labels,cost):
    with tf.name_scope('training'):
        global_step = tf.Variable(0,trainable=False)
        start_learning_rate = 1e-4
        learning_rate = tf.train.exponential_decay(start_learning_rate,global_step,5000,0.96,staircase=True)
        train_step =tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(net,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.scalar('learning_rate',learning_rate)
        return train_step, accuracy

# beginning of the main body

batch_size = 100
total_batch = int(mnist.train_data.data_length/batch_size)
epoch = 10

x = tf.placeholder(tf.float32,[None,28,28,1])
y_  = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

y = inference(x,True)
cross_entropy = cross_entropy(y,y_)
train_step, accuracy = train(y,y_,cross_entropy)

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./summary/train",sess.graph)
test_writer = tf.summary.FileWriter("./summary/test",sess.graph)

sess.run(tf.global_variables_initializer())

# beginning of train step

for i_epoch in range(epoch):
    for i in range(total_batch):
        batch = mnist.train_data.next_batch(batch_size)
        xs = np.reshape(batch[0],(-1,28,28,1))
        ys = batch[1]
        summary, _ = sess.run([merged,train_step],feed_dict = { x: xs, y_: ys, keep_prob: 0.5 })
        train_writer.add_summary(summary,i_epoch*total_batch+i)

        if i % 2000 == 0:
            train_accuracy = accuracy.eval(feed_dict = { x: xs, y_: ys, keep_prob: 1.0})
            print("epoch %d, batch_iteration %d, train_accuracy %g" % (i_epoch,i,train_accuracy))
            print("cross entropy %g" % sess.run(cross_entropy, feed_dict = { x:xs,y_:ys, keep_prob: 1.0}))
            valid_accuracy = accuracy.eval(feed_dict = { x: np.reshape(mnist.valid_data.images,(-1,28,28,1)), y_: mnist.valid_data.labels, keep_prob: 1.0})
            print("validation accuracy %g" % valid_accuracy)

print("test accuracy %g" %  accuracy.eval(feed_dict = { x: np.reshape(mnist.test_data.images,(-1,28,28,1)), y_: mnist.test_data.labels, keep_prob: 1.0}))
            
