# Comparative Analysis of CVAE and CGAN for Image Generation to Improve OCR Systems
This GitHub repository contains code and resources for a comparative analysis of two popular deep learning frameworks, Conditional Variational Autoencoders (CVAE) and Conditional Generative Adversarial Networks (CGAN), for generating synthetic images to enhance Optical Character Recognition (OCR) systems. The goal is to investigate the performance of these models and determine their effectiveness in improving OCR accuracy.
# Results
### GAN
![Real](https://github.com/memari-majid/image_generation/blob/master/real.png)
### GAN
![GAN](https://github.com/memari-majid/image_generation/blob/master/acgan/figs/animation.gif)
### VAE
![VAE](https://github.com/memari-majid/image_generation/blob/master/cvae/figs/animation.gif)
### GAN Loss Functions
<img src="https://github.com/memari-majid/image_generation/blob/master/acgan/loss_plot.png" alt="GAN Loss Function" width="400"/>

### VAE Loss Functions
<img src="https://github.com/memari-majid/image_generation/blob/master/cvae/loss_plot.png" alt="VAE" width="400"/>

### OCR Metrics
<img src="https://github.com/memari-majid/image_generation/blob/master/output/test_accuracy_vs_epoch.png" alt="test accuracy" width="400"/>

### GAN Runtime
<img src="https://github.com/memari-majid/image_generation/blob/master/acgan/elapsed_time_plot.png" alt="GAN" width="400"/>

### VAE Runtime
<img src="https://github.com/memari-majid/image_generation/blob/master/cvae/elapsed_time_plot.png" alt="VAE" width="400"/>

### FID Scores
<img src="https://github.com/memari-majid/image_generation/blob/master/output/fid_scores.png" alt="FID" width="800"/>

# Interpretation
### Generative Models
Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) are two popular generative models used for tasks such as image generation, representation learning, and unsupervised learning. They have different approaches to learning data distributions, which result in different strengths and weaknesses.
### VAEs
VAEs are built on a probabilistic framework and learn an explicit probabilistic representation of the data. They optimize a lower bound on the data likelihood.
They consist of an encoder and a decoder. The encoder maps the input data to a latent space, while the decoder reconstructs the input data from the latent representation.
VAEs are trained using backpropagation and can be optimized with standard gradient-based methods, such as stochastic gradient descent or Adam.
The generated samples from VAEs are generally smoother and have a higher likelihood of being close to the training data distribution, but they can sometimes be blurry or overly smooth.
### GANs:
GANs are based on an adversarial framework where two neural networks, the generator and the discriminator, are trained simultaneously in a zero-sum game.
The generator tries to produce realistic samples, while the discriminator tries to distinguish between real and generated samples. They are trained iteratively to improve each other's performance.
GAN training is often more difficult and unstable, as it requires careful balancing between the generator and discriminator.
The generated samples from GANs tend to be sharper and more visually appealing, but they might not always cover the entire data distribution (i.e., mode collapse).
### VAEs are faster than GANs
Regarding computational speed, the main reason why VAEs can be faster than GANs is their training procedure. VAEs use backpropagation and can be trained with standard optimization algorithms, which leads to a more stable and straightforward training process. In contrast, GANs require the simultaneous training of two networks, and finding the right balance between the generator and discriminator can be challenging. The adversarial training process can be unstable and may require more iterations, hyperparameter tuning, or architectural adjustments to achieve satisfactory results.

### Saliency Maps
<img src="https://github.com/memari-majid/image_generation/blob/master/saliency_maps_cvae.png" alt="Saliency Maps" width="2000"/>

