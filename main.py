import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
import pandas as pd
import argparse
import matplotlib.gridspec as gridspec
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='nb_epoch', type=int, default=5000, help='# of epochs')
parser.add_argument('--learning_rate', dest='lr', type=float, default=0.0001, help='# learning rate')
parser.add_argument('--sample_size', dest='sample_size', type=int, default=60, help='# sample size')
parser.add_argument('--gen_hidden', dest='gen_hidden', type=int, default=80, help='# hidden nodes in generator')
parser.add_argument('--disc_hidden', dest='disc_hidden', type=int, default=80, help='# hidden nodes in discriminator')
parser.add_argument('--your_login', dest='your_login', type=str, default='rubens', help='# your login name')

args = vars(parser.parse_args())

print("\nIMPORTANT NOTICE: Tensorboard will start AFTER you close the pop-up with MNIST digits output.")
var0 = input("Are you using Linux? [y|n]")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

n=args['sample_size']

x_train=x_train[0:n]
x_test=x_test[n:n+n]

noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

def norm(x):
    return(x-np.min(x))/(np.max(x)-np.min(x))

x_train_noisy = norm(x_train_noisy)
x_test_noisy = norm(x_test_noisy)

x_train_noisy=np.concatenate([x_train_noisy,x_train_noisy])
x_test_noisy=np.concatenate([x_test_noisy,x_test_noisy])
x_train=norm(np.concatenate([x_train,x_train]))

np.random.seed(200)
sel=random.sample(range(0,x_train.shape[0]), n)
x_train_noisy=x_train_noisy[sel]
x_test_noisy=x_test_noisy[sel]
x_train=x_train[sel]

y_train=y_train[0:n]
y_train=np.concatenate([y_train,y_train])[sel]
y_train=np.array(pd.get_dummies(y_train)).astype(np.float32)
y_test=y_test[0:n]
y_test0=np.concatenate([y_test,y_test])[sel]
y_test=np.array(pd.get_dummies(y_test0))

x_train=np.array(x_train).astype(np.float64)
x_train_noisy=x_train_noisy.astype(np.float64)

num_steps = args['nb_epoch']
batch_size = args['sample_size']
show_steps=10
learning_rate1=args['lr']
image_dim = 784 
gen_hidden_dim = args['gen_hidden']
disc_hidden_dim = args['disc_hidden']
noise_dim = 10 

num_steps=10

def GAN(sample_size):

    def mean(x):
        mm,_=tf.nn.moments(x,axes=[0])
        return mm
    
    def var(x):
        _,var=tf.nn.moments(x,axes=[0])
        return var
    
    tf.reset_default_graph() 
    def generator(x, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            x = tf.layers.dense(x, units=6 * 6 * 64)
            x = tf.nn.relu(x)
            x = tf.reshape(x, shape=[-1, 6, 6, 64])
            x = tf.layers.conv2d_transpose(x, 32, 4, strides=2)
            x = tf.nn.batch_normalization(x,mean=mean(x), variance=var(x),offset=None,scale=None,variance_epsilon=1e-3)
            x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
            x = tf.nn.relu(x)
            x = tf.reshape(x, [batch_size,784])
            return x
    
    def discriminator(x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            x = tf.reshape(x, [batch_size,28,28,1])
            x = tf.layers.conv2d(x, 32, 5)
            x = tf.nn.relu(x)
            x = tf.layers.average_pooling2d(x, 2, 2,padding='same')
            x = tf.layers.conv2d(x, 64, 5,padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.average_pooling2d(x, 8, 8)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 784)
            x = tf.nn.sigmoid(x)
        return x
    
    
    noise_input = tf.placeholder(tf.float32, shape=[None, 784])
    real_image_input = tf.placeholder(tf.float32, shape=[None, 784])
    
    with tf.name_scope('GenModel'):
        gen_sample = generator(noise_input)
    
    
    disc_real = discriminator(real_image_input)
    
    
    disc_fake = discriminator(gen_sample, reuse=True)
    
    with tf.name_scope('DiscModel'):
        stacked_gan = discriminator(gen_sample, reuse=True)
    
    disc_target = tf.placeholder(tf.float32, shape=[None,784])
    gen_target = tf.placeholder(tf.float32, shape=[None,784])
    
    
    
    with tf.name_scope('GenLoss'):
        gen_loss = tf.reduce_mean(tf.losses.mean_squared_error(
        real_image_input,gen_sample))
        
    
    with tf.name_scope('DiscLoss'):
        disc_loss = tf.reduce_mean(tf.losses.mean_squared_error(
        real_image_input,stacked_gan))
    
    tf.summary.scalar("Generator_Loss", gen_loss)
    tf.summary.scalar("Discriminator_Loss", disc_loss)
    
    if str(var0)=='n':
        logs_path = 'C:/Users/'+args['your_login']+'/Anaconda3/envs/Scripts/plot_1'
    else:
        logs_path = '/home/'+args['your_login']+'/anaconda3/envs/plot_1'
        
    summary = tf.summary.merge_all()
    
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate1)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate1)
       
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    with tf.name_scope('SGDdisc'):
        train_disc = optimizer_disc.minimize(disc_loss)
    
    
    with tf.name_scope('SGDgen'):
        train_gen = optimizer_gen.minimize(gen_loss)
    
    x_image = tf.summary.image('GenSample', tf.reshape(gen_sample, [-1, 28, 28, 1]), 4)
    x_image2 = tf.summary.image('stacked_gan', tf.reshape(stacked_gan, [-1, 28, 28, 1]), 4)
    
    
    for i in range(0,11):
        with tf.name_scope('layer'+str(i)):
            pesos=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            tf.summary.histogram('pesos'+str(i), pesos[i])

  
    summary = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    def next_batch(num, data, labels):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]
        return np.asarray(data_shuffle).astype(np.float32), np.asarray(labels_shuffle).astype(np.float32)
    
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        for i in range(1, num_steps+1):
            batch_x, batch_y=next_batch(sample_size, x_train, x_train_noisy)        
            feed_dict = {real_image_input: batch_x, noise_input: batch_y,
                     disc_target: batch_x, gen_target: batch_y}
            _, _, gl, dl,summary2 = sess.run([train_gen, train_disc, gen_loss, disc_loss,summary],
                                         feed_dict=feed_dict)
            g = sess.run([stacked_gan], feed_dict={noise_input: batch_y})
            h = sess.run([gen_sample], feed_dict={noise_input: batch_y})
            summary_writer.add_summary(summary2, i)
            if i % show_steps == 0 or i == 1:
                print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
        gs = gridspec.GridSpec(3, 8)
        gs.update(wspace=0.5)
        fig=plt.figure(figsize=(10,10))
        for i in range(0,8):
            ax1 = plt.subplot(gs[0, i])
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            plt.title('Orig')
            plt.imshow(x_train_noisy[i].reshape(28, 28))
            plt.gray()
            ax2 = plt.subplot(gs[1, i])
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            plt.title('Gen')
            plt.imshow(np.array(h).reshape(sample_size,28, 28)[i])
            plt.gray()
            ax3 = plt.subplot(gs[2, i])
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            plt.title('Disc')
            plt.imshow(np.array(g).reshape(sample_size,28, 28)[i])
            plt.gray()
            gs.tight_layout(fig)
        plt.show()
        os.system('tensorboard --logdir='+logs_path)
 

            
if __name__ == '__main__':
   GAN(args['sample_size'])
1568/784
