# cppn-gan-vae tensorflow

Train [Compositional Pattern Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) as a Generative Model, using Generative Adversarial Networks and Variational Autoencoder techniques to produce high resolution images.

![Morphing](https://cdn.rawgit.com/hardmaru/cppn-gan-vae-tensorflow/master/examples/output_linear.gif)

Run `python train.py` from the command line to train from scratch and experiment with different settings.

`sampler.py` can be used inside IPython to interactively see results from the models being trained.

See my blog post at [blog.otoro.net](http://blog.otoro.net/2016/04/01/generating-abstract-patterns-with-tensorflow/) for more details.

I tested the implementation on TensorFlow 0.60.

Used images2gif.py written by Almar Klein, Ant1, Marius van Voorden.

# License

BSD - images2gif.py

MIT - everything else
