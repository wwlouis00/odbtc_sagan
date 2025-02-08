from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2
import math

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=4, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None, is_test=False):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
            input_c_dim=3, output_c_dim=3 --> changed 
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model(is_test)

    def build_model(self, is_test):

        if is_test:
            self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, None, None,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
            self.real_B = self.real_data[:, :, :, :self.input_c_dim]
            # print(tf.shape(self.real_B))
            self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

            self.fake_B = self.generator(self.real_A)

            # self.real_AB = tf.concat([self.real_A, self.real_B], 3)
            # self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
            self.real_AB = tf.concat([self.real_A, self.real_B], 3)
            # print(self.real_AB.get_shape())
            self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
            # print(tf.shape(self.fake_AB))
            self.fake_B_sample = self.sampler(self.real_A)

        else:
            print("masuk")
            # self.real_data = tf.placeholder(tf.float32,
            #                                 [self.batch_size, self.image_size, self.image_size,
            #                                  self.input_c_dim + self.output_c_dim],
            #                                 name='real_A_and_B_images')

            self.real_data = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size,
                                             self.input_c_dim + self.output_c_dim],
                                            name='real_A_and_B_images')

            self.real_B = self.real_data[:, :, :, :self.input_c_dim]
            # print(tf.shape(self.real_B))
            self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

            self.fake_B = self.generator(self.real_A)

            # self.real_AB = tf.concat([self.real_A, self.real_B], 3)
            # self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
            self.real_AB = tf.concat([self.real_A, self.real_B], 3)
            # print(self.real_AB.get_shape())
            self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
            # print(tf.shape(self.fake_AB))

            self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
            self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

            # VGG19 Perceptual Loss 
            # # this is for grayscale images
            # self.t_target_vgg = tf.image.resize_images(self.real_AB, size=[224, 224], method=0, align_corners=False)
            # self.t_target_vgg = tf.image.grayscale_to_rgb(self.t_target_vgg[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim])
            # self.t_predict_vgg = tf.image.resize_images(self.fake_AB, size=[224, 224], method=0, align_corners=False)
            # self.t_predict_vgg = tf.image.grayscale_to_rgb(self.t_predict_vgg[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim])
            # self.net_vgg, self.vgg_target_emb = self.vgg19((self.t_target_vgg + 1) / 2)
            # _, self.vgg_predict_emb = self.vgg19((self.t_predict_vgg + 1) / 2)

            # this is for color images
            self.t_target_vgg = tf.image.resize_images(self.real_AB, size=[224, 224], method=0, align_corners=False)
            # self.t_target_vgg = tf.image.grayscale_to_rgb(self.t_target_vgg[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim])
            self.t_predict_vgg = tf.image.resize_images(self.fake_AB, size=[224, 224], method=0, align_corners=False)
            # self.t_predict_vgg = tf.image.grayscale_to_rgb(self.t_predict_vgg[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim])
            self.net_vgg, self.vgg_target_emb = self.vgg19((self.t_target_vgg[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim] + 1) / 2)
            _, self.vgg_predict_emb = self.vgg19((self.t_predict_vgg[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim] + 1) / 2)

            self.fake_B_sample = self.sampler(self.real_A)

            self.d_sum = tf.summary.histogram("d", self.D)
            self.d__sum = tf.summary.histogram("d_", self.D_)
            self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
            self.vgg_loss = 2e-6 * tf.reduce_mean(tf.abs(self.vgg_target_emb - self.vgg_predict_emb))
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) \
                            + self.vgg_loss

            self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
            self.vgg_loss_sum = tf.summary.scalar("vgg_loss", self.vgg_loss)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

            t_vars = tf.trainable_variables()

            self.d_vars = [var for var in t_vars if 'd_' in var.name]
            self.g_vars = [var for var in t_vars if 'g_' in var.name]

            self.saver = tf.train.Saver()
            # print("yuhu")


    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.png'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]

        if (self.is_grayscale):
            # sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            sample_images = np.array(sample).astype(np.float32)
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr / 4, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum, self.vgg_loss_sum])
        # self.g_sum = tf.summary.merge([self.d__sum,
        #     self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        # # NOT NEEDED ! 
        # vgg19_npy_path = 'vgg19.npy'
        # npz = np.load(vgg19_npy_path, encoding='latin1').item()

        # params = []
        # for val in sorted(npz.items()):
        #     W = np.asarray(val[1][0])
        #     b = np.asarray(val[1][1])
        #     params.extend([W,b])
        # tl.files.assign_params(sess, params, net_vgg)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            # print('masuk')
            data = glob('./datasets/{}/train/*.png'.format(self.dataset_name))
            #np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            errD_tot = 0 
            errG_tot = 0

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                if (self.is_grayscale):
                    # batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    batch_images = np.array(batch).astype(np.float32)
                else:
                    batch_images = np.array(batch).astype(np.float32)
                    # print(batch_images.shape)

                # # Update D network
                # _, summary_str = self.sess.run([d_optim, self.d_sum],
                #                                feed_dict={ self.real_data: batch_images })
                # self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                # print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                #     % (epoch, idx, batch_idxs,
                #         time.time() - start_time, errD_fake+errD_real, errG))

                errD_tot = errD_tot + errD_real + errD_fake
                errG_tot = errG_tot + errG

                # for color images, change the counter to 1233. for grayscale images, change the counter to 1234.
                if np.mod(counter, 1233) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 1233) == 1:
                    self.save(args.checkpoint_dir, counter)

            errD_tot = errD_tot / (len(data) / self.batch_size)
            errG_tot = errG_tot / (len(data) / self.batch_size)
            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, time.time() - start_time, errD_tot, errG_tot))

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            # print(h4.get_shape())

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # print(e1.get_shape())
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            # coba attention setelah deconvolution, sebelum dropout 
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = self.attention(self.d1, self.gf_dim*8, name='att_d1')
            # d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.nn.dropout(self.g_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = self.attention(self.d2, self.gf_dim*8, name='att_d2')
            # d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.nn.dropout(self.g_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = self.attention(self.d3, self.gf_dim*8, name='att_d3')
            # d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.nn.dropout(self.g_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.attention(self.d4, self.gf_dim*8, name='att_d4')
            # d4 = self.g_bn_d4(self.d4)
            d4 = self.g_bn_d4(d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.attention(self.d5, self.gf_dim*4, name='att_d5')
            # d5 = self.g_bn_d5(self.d5)
            d5 = self.g_bn_d5(d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.attention(self.d6, self.gf_dim*2, name='att_d6')
            # d6 = self.g_bn_d6(self.d6)
            d6 = self.g_bn_d6(d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            # d7 = self.attention(self.d7, self.gf_dim*1, name='att_d7')
            d7 = self.g_bn_d7(self.d7)
            # d7 = self.g_bn_d7(d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)


    # attention starts here 
    def attention(self, x, ch, name="attention"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            # f = tf.layers.conv2d(x, ch // 8, kernel_size=1, strides=1, name='f_conv')
            # g = tf.layers.conv2d(x, ch // 8, kernel_size=1, strides=1, name='g_conv')
            f = tf.layers.conv2d(x, 32, kernel_size=1, strides=1, name='f_conv')
            g = tf.layers.conv2d(x, 32, kernel_size=1, strides=1, name='g_conv')
            h = tf.layers.conv2d(x, ch, kernel_size=1, strides=1, name='h_conv')

            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)

            beta = tf.nn.softmax(s)

            o = tf.matmul(beta, hw_flatten(h))

            gamma = tf.get_variable('gamma', [1], initializer=tf.constant_initializer(0.0))

            # o = tf.reshape(o, shape=x.shape)
            o = tf.reshape(o, shape=tf.shape(x))
            x = gamma * o + x 

        return x

    def attention_2(self, x, ch):
        with tf.variable_scope("attention_2", reuse=tf.AUTO_REUSE) as scope:
            flat_x = hw_flatten(x)

            f = tf.layers.conv1d(flat_x, 8, kernel_size=1, name='f_conv')
            g = tf.layers.conv1d(flat_x, 8, kernel_size=1, name='g_conv')
            h = tf.layers.conv1d(flat_x, ch, kernel_size=1, name='h_conv')

            beta = tf.nn.softmax(tf.matmul(f, g, transpose_b=True))
            o = tf.matmul(beta, h)

            gamma = tf.get_variable('gamma', [], initializer=tf.zeros_initializer)
            y = gamma * o + flat_x
            y = tf.reshape(y, x.shape)
        return y

    def vgg19(self, rgb):
        # print(rgb.get_shape())
        vgg19_npy_path = 'vgg19.npy'
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()

        VGG_MEAN = [103.939, 116.779, 123.68]
        with tf.variable_scope('vgg19', reuse=tf.AUTO_REUSE) as vs:
            rgb_scaled = rgb * 255.0
            rgb_scaled = tf.cast(rgb_scaled, tf.float32)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            if tf.__version__ <= '0.11':
                bgr = tf.concat(3, [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2]
                ])
            else:
                bgr = tf.concat([
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    blue - VGG_MEAN[2]
                ], axis=3)
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
            # print(bgr.get_shape())

            # conv 1 
            network = tf.layers.conv2d(bgr, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv1_1')
            # print(network.get_shape())
            network = tf.layers.conv2d(network, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv1_2')
            network = tf.layers.max_pooling2d(network, pool_size=2, strides=2, padding='same', name='vggpool1')

            # conv 2 
            network = tf.layers.conv2d(network, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv2_1')
            network = tf.layers.conv2d(network, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv2_2')
            network = tf.layers.max_pooling2d(network, pool_size=2, strides=2, padding='same', name='vggpool2')

            # conv 3 
            network = tf.layers.conv2d(network, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv3_1')
            network = tf.layers.conv2d(network, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv3_2')
            network = tf.layers.conv2d(network, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv3_3')
            network = tf.layers.conv2d(network, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv3_4')
            network = tf.layers.max_pooling2d(network, pool_size=2, strides=2, padding='same', name='vggpool3')

            # conv 4 
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv4_1')
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv4_2')
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv4_3')
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv4_4')
            network = tf.layers.max_pooling2d(network, pool_size=2, strides=2, padding='same', name='vggpool4')
            conv = network

            # conv 5 
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv5_1')
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv5_2')
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv5_3')
            network = tf.layers.conv2d(network, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='vggconv5_4')
            network = tf.layers.max_pooling2d(network, pool_size=2, strides=2, padding='same', name='vggpool5')

            # fc 6-8
            network = tf.layers.flatten(network, name='vggflatten')
            network = tf.layers.dense(network, units=4096, activation=tf.nn.relu, name='vggfc6')
            network = tf.layers.dense(network, units=4096, activation=tf.nn.relu, name='vggfc7')
            network = tf.layers.dense(network, units=1000, activation=tf.identity, name='vggfc8')
            return network, conv


    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            # s = 300

            # print(s)
            # if s != 256 or s!= 512:
            #     s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            # else:
            #     s2, s4, s8, s16, s32, s64, s128 = int(s/2 + 1), int(s/4 + 1), int(s/8 + 1), int(s/16 + 1), int(s/32 + 1), int(s/64 + 1), int(s/128 + 1)
            #     print(s128)

            # s2, s4, s8, s16, s32, s64, s128 = int(round(s/2)), int(round(s/4)), int(round(s/8)), int(round(s/16)), int(round(s/32)), int(round(s/64)), int(round(s/128))
            # s2, s4, s8, s16, s32, s64, s128 = round(s/2), round(s/4), round(s/8), round(s/16), round(s/32), round(s/64), round(s/128)
            height, width = image.shape[1:3]
            # print(height)

            # # comment this in training phase 
            # if height >= 0:
            #     print('masu')
            #     s_h = int(math.ceil(height/2))
            #     s2_h = int(math.ceil(s_h/2))
            #     s4_h = int(math.ceil(s2_h/2))
            #     s8_h = int(math.ceil(s4_h/2))
            #     s16_h = int(math.ceil(s8_h/2))
            #     s32_h = int(math.ceil(s16_h/2))
            #     s64_h = int(math.ceil(s32_h/2))
            #     s128_h = int(math.ceil(s64_h/2))

            #     s_w = int(math.ceil(width/2))
            #     s2_w = int(math.ceil(s_w/2))
            #     s4_w = int(math.ceil(s2_w/2))
            #     s8_w = int(math.ceil(s4_w/2))
            #     s16_w = int(math.ceil(s8_w/2))
            #     s32_w = int(math.ceil(s16_w/2))
            #     s64_w = int(math.ceil(s32_w/2))
            #     s128_w = int(math.ceil(s64_w/2))

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # print(e3.get_shape())
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            dyn_input_shape = tf.shape(e7)
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim*8], name='g_d1', with_w=True)
            d1 = self.attention(self.d1, self.gf_dim*8, name='att_d1')
            # d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.nn.dropout(self.g_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            dyn_input_shape = tf.shape(e6)
            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim*8], name='g_d2', with_w=True)
            d2 = self.attention(self.d2, self.gf_dim*8, name='att_d2')
            # d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.nn.dropout(self.g_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            dyn_input_shape = tf.shape(e5)
            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim*8], name='g_d3', with_w=True)
            d3 = self.attention(self.d3, self.gf_dim*8, name='att_d3')
            # d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.nn.dropout(self.g_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            dyn_input_shape = tf.shape(e4)
            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.attention(self.d4, self.gf_dim*8, name='att_d4')
            # d4 = self.g_bn_d4(self.d4)
            d4 = self.g_bn_d4(d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            dyn_input_shape = tf.shape(e3)
            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.attention(self.d5, self.gf_dim*4, name='att_d5')
            # d5 = self.g_bn_d5(self.d5)
            d5 = self.g_bn_d5(d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            dyn_input_shape = tf.shape(e2)
            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.attention(self.d6, self.gf_dim*2, name='att_d6')
            # d6 = self.g_bn_d6(self.d6)
            d6 = self.g_bn_d6(d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            dyn_input_shape = tf.shape(e1)
            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.gf_dim], name='g_d7', with_w=True)
            # d7 = self.attention(self.d7, self.gf_dim*1, name='att_d7')
            d7 = self.g_bn_d7(self.d7)
            # d7 = self.g_bn_d7(d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            dyn_input_shape = tf.shape(image)
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, dyn_input_shape[1], dyn_input_shape[2], self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, is_test=False):
        if is_test:
            print(" [*] Reading checkpoint...")

            temp_saver = tf.train.Saver()
            model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
            checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

            variables_can_be_restored = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            # print(variables_can_be_restored)
            temp_saver = tf.train.Saver(variables_can_be_restored)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                temp_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
        else:
            print(" [*] Reading checkpoint...")

            temp_saver = tf.train.Saver()
            model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
            checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/test/*.png'.format(self.dataset_name))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.png')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample_images = [load_data(sample_file, is_test=True) for sample_file in sample_files]
        # print(sample.shape)

        # # if color, pls comment these lines 
        # sampleArr = np.array(sample_images)
        # # print(sampleArr[0].shape)
        # for i in range(0,len(sample_images)):
        #     # pic = sampleArr[i,:,:,0]
        #     pic = sampleArr[i]
        #     pic = pic[:,:,0:3]
        #     pic = cv2.normalize(pic, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        #     pic = pic.astype(np.uint8)
        #     cv2.imwrite('./{}/test_real_{:04d}.png'.format(args.test_dir, i+1),pic)
        # # pic = sampleArr[0,:,:,0]
        # # print(sampleArr[0,:,:,0])
        # # print(pic.type)
        # # end of pls comment these lines

        # if (self.is_grayscale):
        #     # sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        #     sample_images = np.array(sample).astype(np.float32)
        # else:
        #     sample_images = np.array(sample).astype(np.float32)

        # sample_images = [sample_images[i:i+self.batch_size]
        #                  for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        # print(sample_images[0])

        start_time = time.time()
        if self.load(self.checkpoint_dir, is_test=True):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # for i, sample_image in enumerate(sample_images):
        #     idx = i+1
        #     print("sampling image ", idx)
        #     # realImg = sample_image[:,:,:,0]
        #     # print(realImg)
        #     # cv2.imwrite('./{}/test_real_{:04d}.png'.format(args.test_dir, idx),sample_image[:,:,:,0])
        #     sample_image = np.asarray(sample_image)
        #     print(sample_image.shape)

        #     samples = self.sess.run(
        #         self.fake_B_sample,
        #         feed_dict={self.real_data: sample_image}
        #     )
        #     save_images(samples, [self.batch_size, 1],
        #                 './{}/test_gen_{:04d}.png'.format(args.test_dir, idx))

        for i in range(0, len(sample_images)):
            idx = i + 1
            print("sampling image : ", idx)

            sample_image = sample_images[i]
            sample_image = np.expand_dims(sample_image, axis = 0)
            print(sample_image.shape)

            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        './{}/test_gen_{:04d}.png'.format(args.test_dir, idx))

