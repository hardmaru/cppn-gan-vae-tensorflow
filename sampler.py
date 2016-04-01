'''
Implementation of Compositional Pattern Producing Networks in Tensorflow

https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

@hardmaru, 2016

Sampler Class

This file is meant to be run inside an IPython session, as it is meant
to be used interacively for experimentation.

It shouldn't be that hard to take bits of this code into a normal
command line environment though if you want to use outside of IPython.

usage:

%run -i sampler.py

sampler = Sampler()

'''

import numpy as np
import tensorflow as tf
import math
import random
import PIL
from PIL import Image
import pylab
from model import CPPNVAE
import matplotlib.pyplot as plt
import images2gif
from images2gif import writeGif

mgc = get_ipython().magic
mgc(u'matplotlib inline')
mgc(u'run -i mnist_data.py')
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

class Sampler():
  def __init__(self):
    self.mnist = None
    self.model = CPPNVAE()
    self.model.load_model('save')
    self.z = self.generate_z()
  def get_random_mnist(self, with_label = False):
    if self.mnist == None:
      self.mnist = read_data_sets()
    if with_label == True:
      data, label = self.mnist.next_batch(1, with_label)
      return data[0], label[0]
    return self.mnist.next_batch(1)[0]
  def get_random_specific_mnist(self, label = 2):
    m, l = self.get_random_mnist(with_label = True)
    for i in range(100):
      if l == label:
        break
      m, l = self.get_random_mnist(with_label = True)
    return m
  def generate_random_label(self, label):
    m = self.get_random_specific_mnist(label)
    self.show_image(m)
    self.show_image_from_z(self.encode(m))
  def generate_z(self):
    z = np.random.normal(size=self.model.z_dim).astype(np.float32)
    return z
  def encode(self, mnist_data):
    new_shape = [1]+list(mnist_data.shape)
    return self.model.encode(np.reshape(mnist_data, new_shape))
  def generate(self, z=None, x_dim=512, y_dim=512, scale = 8.0):
    if z is None:
      z = self.generate_z()
    else:
      z = np.reshape(z, (1, self.model.z_dim))
    self.z = z
    return self.model.generate(z, x_dim, y_dim, scale)[0]
  def show_image(self, image_data):
    '''
    image_data is a tensor, in [height width depth]
    image_data is NOT the PIL.Image class
    '''
    plt.subplot(1, 1, 1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.model.c_dim
    if c_dim > 1:
      plt.imshow(image_data, interpolation='nearest')
    else:
      plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()
  def show_image_from_z(self, z):
    self.show_image(self.generate(z))
  def save_png(self, image_data, filename, specific_size = None):
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.model.c_dim
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    if specific_size != None:
      im = im.resize(specific_size)
    im.save(filename)
  def to_image(self, image_data):
    # convert to PIL.Image format from np array (0, 1)
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.model.c_dim
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    return im
  def morph(self, z1, z2, n_total_frame = 10, x_dim = 512, y_dim = 512, scale = 8.0, sinusoid = False):
    '''
    returns a list of img_data to represent morph between z1 and z2
    default to linear morph, but can try sinusoid for more time near the anchor pts
    n_total_frame must be >= 2, since by definition there's one frame for z1 and z2
    '''
    delta_z = 1.0 / (n_total_frame-1)
    diff_z = (z2-z1)
    img_data_array = []
    for i in range(n_total_frame):
      percentage = delta_z*float(i)
      factor = percentage
      if sinusoid == True:
        factor = np.sin(percentage*np.pi/2)
      z = z1 + diff_z*factor
      print "processing image ", i
      img_data_array.append(self.generate(z, x_dim, y_dim, scale))
    return img_data_array
  def save_anim_gif(self, img_data_array, filename, duration = 0.1):
    '''
    this saves an animated gif given a list of img_data (numpy arrays)
    '''
    images = []
    for i in range(len(img_data_array)):
      images.append(self.to_image(img_data_array[i]))
    writeGif(filename, images, duration = duration)
