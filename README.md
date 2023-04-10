### Comparative Analysis of CVAE and CGAN for Image Generation to Improve OCR Systems
This GitHub repository contains code and resources for a comparative analysis of two popular deep learning frameworks, namely Conditional Variational Autoencoders (CVAE) and Conditional Generative Adversarial Networks (CGAN), for generating synthetic images to enhance Optical Character Recognition (OCR) systems. The goal is to investigate the performance of these models and determine their effectiveness in improving OCR accuracy.

### Project Structure
The project is organized into several Python scripts and Jupyter notebooks, including:

acgan and vae directories, which contain files for training the CVAE and CGAN models, respectively.
main, which defines the classification model architecture and training functions.
results, which contains generated images, classification metrics, and Saliency Maps for interpreting the classification.
README.md, which provides the project description and setup instructions.
Setup and Installation
To run this project, follow these steps:

Clone the repository by running git clone https://github.com/yourusername/Arabic-Handwritten-Digit-Recognition-with-GANs.git in your command line interface.
Install the required dependencies by running pip install -r requirements.txt.
Obtain a Kaggle API key by following the instructions in Kaggle's documentation.
Results
The research results are presented in the form of generated plots and console output. These include comparisons of synthetic image quality, recognition performance improvements, and classifier robustness against adversarial attacks.

![GIF](https://github.com/memari-majid/image_generation/blob/master/acgan/figs/animation.gif)

### Acknowledgments
We would like to express our gratitude to Dr. Lo'ai Ali Tawalbeh and his team for providing us with the Arabic Handwritten Digits Dataset. We also acknowledge the foundational work of Ian Goodfellow and his colleagues in their paper on Generative Adversarial Networks and Diederik P Kingma and Max Welling for their work on Auto-Encoding Variational Bayes.
