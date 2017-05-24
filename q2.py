# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:51:24 2016

@author: chaoran
"""

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
 
#Convolution and Pooling
#convolutions uses a stride of one and are zero padded 
#so that the output is the same size as the input.  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#max pooling over 2x2 blocks. 
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#First Convolutional Layer 
#The convolutional will compute 32 features for each 5x5 patch. 
#Its weight tensor will have a shape of [5, 5, 1, 32]. 
#The first two dimensions are the patch size, the next is 
#the number of input channels, and the last is the number of output channels. 
#We will also have a bias vector with a component for each output channel.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#To apply the layer, we first reshape x to a 4d tensor, with the second and 
#third dimensions corresponding to image width and height, and the final 
#dimension corresponding to the number of color channels
x_image = tf.reshape(x, [-1,28,28,1])
#an averaging a square of 4x4 pixels 
#thus transforming the 28x28 images into 7x7 images.
#strides: shifting the filter by 4x4 pixels
x_image = tf.nn.avg_pool(x_image, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')
#convolve x_image with the weight tensor, add the bias, 
#apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#result in 4x4 images

#Second Convolutional Layer
#The second layer will have 64 features for each 3x3 patch.
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#result in 2x2 images

#Densely Connected Layer
#Now that the image size has been reduced to 2x2, 
#we add a fully-connected layer with 1024 neurons to allow processing on 
#the entire image. We reshape the tensor from the pooling layer into 
#a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([2 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
#To reduce overfitting, we will apply dropout before the readout layer. 
#We create a placeholder for the probability that a neuron's output is kept 
#during dropout. This allows us to turn dropout on during training, 
#and turn it off during testing. 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
train_a = []
valid_a = []
a = []
for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    valid_d = accuracy.eval(feed_dict={
        x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
    train_a.append(train_accuracy)
    valid_a.append(valid_d)
    a.append(i)
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



#training errow is red
#vaildation errow is blue
plt.plot(a, train_a,'r-',label='training error')
plt.plot(a, valid_a, 'b-', label='validation error')
plt.xlabel("steps")
plt.ylabel("percentage accuracy")
plt.legend(loc=4)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#for i in range(10):
    #testSet = mnist.test.next_batch(50)
    #test_accuracy = accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0})
    #print("test accuracy %g"%(test_accuracy))
    
plt.show()