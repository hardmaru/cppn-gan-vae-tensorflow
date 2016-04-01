"""Functions for downloading and reading MNIST data."""
'''
modified for the purpose of cppgan demo
'''
import gzip
import os
import urllib
import numpy as np
import matplotlib.pyplot as plt
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath
def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)
def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

# class to store mnist data
class DataSet(object):
  def __init__(self, images, labels):
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._num_examples = len(images)
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self._images = images[perm]
    self._labels = labels[perm]
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, with_label = False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    if with_label == True:
      return self.distort_batch(self._images[start:end]), self._labels[start:end]
    return self.distort_batch(self._images[start:end])
  def distort_batch(self, batch):
    batch_size = len(batch)
    row_distort = np.random.randint(0, 3, batch_size)
    col_distort = np.random.randint(0, 3, batch_size)
    result = np.zeros(shape=(batch_size, 26, 26, 1), dtype=np.float32)
    for i in range(batch_size):
      result[i, :, :, :] = batch[i, row_distort[i]:row_distort[i]+26, col_distort[i]:col_distort[i]+26, :]
    return result
  def show_image(self, image):
    plt.subplot(1, 1, 1)
    plt.imshow(np.reshape(image, (26, 26)), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()
  def shuffle_data(self):
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self._images = self._images[perm]
    self._labels = self._labels[perm]

def read_data_sets(train_dir = 'MNIST_data', one_hot=False):
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  all_images = np.vstack((train_images, test_images))
  all_labels = np.concatenate((train_labels, test_labels))

  #data_sets = DataSet(all_images, all_labels) # train on all train+test sets 70k
  data_sets = DataSet(train_images, train_labels) # train only only train set 60k
  return data_sets

