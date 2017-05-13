#!/usr/bin/python3

from mnist import MnistData
import tensorflow as tf
import numpy as np

pickle_file = 'notMNIST_sanit.pickle'
mnist = MnistData(pickle_file,one_hot=True)

sess = tf.InteractiveSession()

batch_size = 100
total_batch = int(mnist.train_data.data_length/batch_size)
epoch = 10
learning_rate = 1e-4

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# variable

x = tf.placeholder(tf.float32,[None,28,28,1])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

# first conv layer
with tf.name_scope("layer1"):
    W_conv1 = weight_variable([4,4,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# second conv layer
with tf.name_scope("layer2"):
    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# third conv layer
with tf.name_scope("layer3"):
    W_conv3 = weight_variable([5,5,64,128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)

# FC layer

print('fuck fuck fuck')
print(h_conv3.shape)

with tf.name_scope("layer-4-fc"):
    W_fc1 = weight_variable([7*7*128,1024])
    b_fc1 = bias_variable([1024])
    h_conv3_flat = tf.reshape(h_conv3,[-1,7*7*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)

# merge all the summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./summary/train",sess.graph)
test_writer = tf.summary.FileWriter("./summary/test",sess.graph)
    
sess.run(tf.global_variables_initializer())

for iteration in range(epoch):
    for i in range(total_batch):
        batch = mnist.train_data.next_batch(batch_size)
        xs = np.reshape(batch[0],(-1,28,28,1))
        ys = batch[1]
        summary, _ = sess.run([merged,train_step],feed_dict = { x: xs, y_: ys, keep_prob: 0.75})
        train_writer.add_summary(summary,i)

        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict = { x: xs, y_: ys, keep_prob: 1.0})
            print("epoch: %d, batch iteration: %d, train accuracy %g" % (iteration,i,train_accuracy))
            
            print("Cross Entropy ",sess.run(cross_entropy,feed_dict = { x: xs, y_: ys, keep_prob: 1.0}))
            print("validation accuracy %g" % accuracy.eval( feed_dict = { x: np.reshape(mnist.valid_data.images,(-1,28,28,1)), y_: mnist.valid_data.labels, keep_prob: 1.0 }))

            
print("test accuracy %g" % accuracy.eval( feed_dict = { x: np.reshape(mnist.test_data.images,(-1,28,28,1)), y_: mnist.test_data.labels, keep_prob: 1.0 }))


