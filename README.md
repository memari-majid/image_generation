# Comparative Analysis of VAE and GAN for Image Generation to Improve OCR Systems
This GitHub repository contains code and resources for a comparative analysis of two popular deep learning frameworks, Conditional Variational Autoencoders (CVAE) and Conditional Generative Adversarial Networks (CGAN), for generating synthetic images to enhance Optical Character Recognition (OCR) systems. The goal is to investigate the performance of these models and determine their effectiveness in improving OCR accuracy.
# Results
### Real
![Real](https://github.com/memari-majid/image_generation/blob/master/real.png)
### GAN
![GAN](https://github.com/memari-majid/image_generation/blob/master/acgan/figs/animation.gif)
### VAE
![VAE](https://github.com/memari-majid/image_generation/blob/master/cvae/figs/animation.gif)

<table>
  <tr>
    <td>
      <h3>GAN Loss Functions</h3>
      <img src="https://github.com/memari-majid/image_generation/blob/master/acgan/loss_plot.png" alt="GAN Loss Function" width="400"/>
    </td>
    <td>
      <h3>VAE Loss Functions</h3>
      <img src="https://github.com/memari-majid/image_generation/blob/master/cvae/loss_plot.png" alt="VAE" width="400"/>
    </td>
  </tr>
</table>


### OCR Metrics
<table>
  <tr>
    <td><img src="https://github.com/memari-majid/image_generation/blob/master/output/accuracy.png" alt="test accuracy" width="400"/></td>
    <td><img src="https://github.com/memari-majid/image_generation/blob/master/output/F1.png" alt="test accuracy" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/memari-majid/image_generation/blob/master/output/precision.png" alt="precision" width="400"/></td>
    <td><img src="https://github.com/memari-majid/image_generation/blob/master/output/recall.png" alt="recall" width="400"/></td>
  </tr>
</table>


<table>
  <tr>
    <td>
      <h3>GAN Runtime</h3>
      <img src="https://github.com/memari-majid/image_generation/blob/master/acgan/elapsed_time_plot.png" alt="GAN" width="400"/>
    </td>
    <td>
      <h3>VAE Runtime</h3>
      <img src="https://github.com/memari-majid/image_generation/blob/master/cvae/elapsed_time_plot.png" alt="VAE" width="400"/>
    </td>
  </tr>
</table>


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

# Conclusion
In conclusion, our study found that the Variational Autoencoder (VAE) model is more suitable for enhancing real-time OCR performance compared to Generative Adversarial Networks (GANs). This is primarily due to the speed of the VAE model, which is 10 times faster than GANs in generating synthetic images.

Our study also introduced a novel evaluation metric, Low-dimensional Fréchet Inception Distance (LFID), which proved to be an accurate and efficient alternative to traditional FID scores for real-time monitoring of image generation. Additionally, our analysis utilizing Saliency Maps demonstrated that the improvement in OCR performance is valid because the OCR system is utilizing unique features of the digits for classification.

Overall, the proposed approach of combining generative-based data augmentation techniques with novel evaluation metrics like LFID can significantly improve OCR performance in real-time applications, capable of handling challenges such as noise, distortions, and limited availability of training data. These findings can pave the way for the development of more advanced OCR systems, capable of handling a broader range of applications.

### Saliency Maps
To interpret the output maps generated by the saliency map technique Grad-CAM, one should look for regions of the input image that are highlighted in the heatmap, as these correspond to the parts of the image that the model considered most important for its decision. Hotter colors on the heatmap represent higher importance, while cooler colors represent lower importance. However, saliency maps should be interpreted with caution as they are not always reliable indicators of how a model is making its predictions, and they only show which parts of the image are important for the model's decision, but not why or how they are being used. Therefore, they should be used in conjunction with other methods of model interpretation and validation.
<img src="https://github.com/memari-majid/image_generation/blob/master/saliency_maps_cvae.png" alt="Saliency Maps" width="2000"/>

