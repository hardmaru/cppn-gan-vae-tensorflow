import argparse
import os
import time

import numpy as np
import tensorflow as tf
from six import BytesIO, StringIO
from six.moves import cPickle

import dataset
from model import CPPNVAE

"""
cppn vae:

compositional pattern-producing generative adversarial network

LOADS of help was taken from:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

"""

parser = argparse.ArgumentParser()
parser.add_argument("--training_epochs", type=int, default=100, help="training epochs")
parser.add_argument("--display_step", type=int, default=1, help="display step")
parser.add_argument("--checkpoint_step", type=int, default=1, help="checkpoint step")
parser.add_argument("--batch_size", type=int, default=500, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.005, help="learning rate for G and VAE"
)
parser.add_argument(
    "--learning_rate_vae", type=float, default=0.001, help="learning rate for VAE"
)
parser.add_argument(
    "--learning_rate_d", type=float, default=0.001, help="learning rate for D"
)
parser.add_argument(
    "--keep_prob", type=float, default=1.00, help="dropout keep probability"
)
parser.add_argument(
    "--beta1", type=float, default=0.65, help="adam momentum param for descriminator"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=dataset.SAVE_MNIST,
    help="output dir for model checkpoint files",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=dataset.DIR_MNIST,
    help="output dir for model checkpoint files",
)


def main():
    args = parser.parse_args()
    return train(args)


"""
usage (in jupyter):

```
  from train import parser, train

  args = parser.parse_args("--save_dir save-job-001".split())
  train(args)
```
"""


def to_image(data):
    # convert to PIL.Image format from np array (0, 1)
    img_data = np.array(1 - data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.model.c_dim
    if c_dim > 1:
        img_data = np.array(
            img_data.reshape((y_dim, x_dim, c_dim)) * 255.0, dtype=np.uint8
        )
    else:
        img_data = np.array(img_data.reshape((y_dim, x_dim)) * 255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    return im


def train(args):
    learning_rate = args.learning_rate
    learning_rate_d = args.learning_rate_d
    learning_rate_vae = args.learning_rate_vae
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    checkpoint_step = (
        args.checkpoint_step
    )  # save training results every check point step
    beta1 = args.beta1
    keep_prob = args.keep_prob

    data_dir = args.data_dir

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        cPickle.dump(args, f)
    checkpoint_path = os.path.join(save_dir, "model.ckpt")

    mnist_dataset = dataset.read_data_sets(train_dir=data_dir)
    n_samples = mnist_dataset.num_examples

    cppnvae = CPPNVAE(
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_d=learning_rate_d,
        learning_rate_vae=learning_rate_vae,
        beta1=beta1,
        keep_prob=keep_prob,
    )

    # load previously trained model if appilcable
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt:
        cppnvae.load_model(save_dir)

    counter = 0

    sample_img = None

    # Training cycle
    for epoch in range(training_epochs):
        avg_d_loss = 0.0
        avg_q_loss = 0.0
        avg_vae_loss = 0.0
        mnist_dataset.shuffle_data()
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_images = mnist_dataset.next_batch(batch_size)
            if not sample_img:
                sample_data = batch_images[0]
                print(sample_data)
                sample_img = to_image(sample_data)
                print(sample_img)
                # Write the image to a string
                try:
                    s = StringIO()
                    sample_img.save(s, "PNG")
                except:
                    s = BytesIO()
                    sample_img.save(s, "PNG")

                # Create an Image object
                img_sum = tf.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=img.shape[0],
                    width=img.shape[1],
                )
                cppnvae.writer.add_summary(
                    tf.Summary(value=tf.Summary.Value(tag="sample_img", image=img_sum))
                )
                cppnvae.writer.flush()

            d_loss, g_loss, vae_loss, n_operations = cppnvae.partial_train(batch_images)

            assert vae_loss < 1000000  # make sure it is not NaN or Inf
            assert d_loss < 1000000  # make sure it is not NaN or Inf
            assert g_loss < 1000000  # make sure it is not NaN or Inf

            # Display logs per epoch step
            if (counter + 1) % display_step == 0:
                print(
                    "Sample:",
                    "%d" % ((i + 1) * batch_size),
                    " Epoch:",
                    "%d" % (epoch),
                    "d_loss=",
                    "{:.4f}".format(d_loss),
                    "g_loss=",
                    "{:.4f}".format(g_loss),
                    "vae_loss=",
                    "{:.4f}".format(vae_loss),
                    "n_op=",
                    "%d" % (n_operations),
                )
            counter += 1
            # Compute average loss
            avg_d_loss += d_loss / n_samples * batch_size
            avg_q_loss += g_loss / n_samples * batch_size
            avg_vae_loss += vae_loss / n_samples * batch_size

        # Display logs per epoch step
        if epoch >= 0:
            print(
                "Epoch:",
                "%04d" % (epoch),
                "avg_d_loss=",
                "{:.6f}".format(avg_d_loss),
                "avg_q_loss=",
                "{:.6f}".format(avg_q_loss),
                "avg_vae_loss=",
                "{:.6f}".format(avg_vae_loss),
            )
            metrics = dict(
                avg_d_loss=avg_d_loss, avg_q_loss=avg_q_loss, avg_vae_loss=avg_vae_loss
            )
            for key in sorted(metrics):
                summary = tf.Summary(
                    value=[
                        tf.Summary.Value(tag=key, simple_value=metrics[key]),
                    ]
                )
                cppnvae.writer.add_summary(summary, epoch)
            cppnvae.writer.flush()

        # save model
        if epoch >= 0 and epoch % checkpoint_step == 0:
            cppnvae.save_model(checkpoint_path, epoch)
            print("model saved to {}".format(checkpoint_path))

        # save model one last time, under zero label to denote finish.
        cppnvae.save_model(checkpoint_path, 0)
    cppnvae.writer.close()


if __name__ == "__main__":
    main()
