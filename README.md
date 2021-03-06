# VAE-MNIST

Autoencoders are a type of neural network that can be used to learn efficient codings of input data. 

An autoencoder network is actually a pair of two connected networks, an encoder and a decoder. An encoder network takes in an input, and converts it into a smaller, dense representation, which the decoder network can use to convert it back to the original input.

Standard autoencoders learn to generate compact representations and reconstruct their inputs well, but asides from a few applications like denoising autoencoders, they are fairly limited.

However, there are much more interesting applications for autoencoders.

One such application is called the variational autoencoder. Using variational autoencoders, it’s not only possible to compress data — it’s also possible to generate new objects of the type the autoencoder has seen before.

Using a general autoencoder, we don’t know anything about the coding that’s been generated by our network. We could compare different encoded objects, but it’s unlikely that we’ll be able to understand what’s going on. This means that we won’t be able to use our decoder for creating new objects. We simply don’t know what the inputs should look like.

Using a variational autoencoder, we take the opposite approach. We will not try to make guesses concerning the distribution that’s being followed by the latent vectors. We simply tell our network what we want this distribution to look like.

Usually, we will constrain the network to produce latent vectors having entries that follow the unit normal distribution. Then, when trying to generate data, we can simply sample some values from this distribution, feed them to the decoder, and the decoder will return us completely new objects that appear just like the objects our network has been trained with.

This project was part of my coursework as a graduate student.

## References:

1. https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
2. https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
3. http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Spring.2018/www/slides/lec16.vae.pdf
4. http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
5. https://www.ics.uci.edu/~welling/publications/papers/AEVB_ICLR14.pdf