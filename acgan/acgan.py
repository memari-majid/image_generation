# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnsit
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import gpustat
from tensorflow.keras.callbacks import TensorBoard
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
import pandas as pd
import os
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal

import tensorflow as tf
import keras
from keras.initializers import RandomNormal
import time
import psutil
from gpustat import GPUStatCollection
import imageio

# define the standalone discriminator model


def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
    # weight initialization
    init = RandomNormal(mean=0.0, stddev=0.02, seed=42)
    # image input
    in_image = Input(shape=in_shape)
    # downsample to 14x14
    fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # downsample to 7x7
    fe = Conv2D(128, (3, 3), strides=(2, 2),
                padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = Adam(learning_rate=0.001, epsilon=1e-08)
    model.compile(loss=['binary_crossentropy',
                  'sparse_categorical_crossentropy'], optimizer=opt)
    return model

# define the standalone generator model


def define_generator(latent_dim, n_classes=10):
    # weight initialization
    init = RandomNormal(mean=0.0, stddev=0.02, seed=42)
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 384 * 7 * 7
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((7, 7, 384))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(192, (5, 5), strides=(
        2, 2), padding='same', kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(1, (5, 5), strides=(
        2, 2), padding='same', kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(learning_rate=0.001, epsilon=1e-08)
    model.compile(loss=['binary_crossentropy',
                  'sparse_categorical_crossentropy'], optimizer=opt)
    return model


def load_real_samples():
    # Load the data
    X_train = np.load(
        '/home/siu853655961/image_generation/data/train_images.npy')
    y_train = np.load(
        '/home/siu853655961/image_generation/data/train_labels.npy')
    # expand to 3d, e.g. add channels dimension
    X = X_train.reshape(-1, 28, 28, 1)
    # flip the images vertically
    X = np.flip(X, axis=2)
    # rotate the images 90 degrees clockwise
    X = np.rot90(X, axes=(1, 2))
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    # load the class labels
    y = y_train
    return X, y


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# genrate and save images and lables
def generate(epoch, g_model, latent_dim, output_dir):
    num_labels = 10
    num_images = 1000
    noise = np.random.randn(num_labels * num_images, latent_dim)
    labels = np.repeat(np.arange(num_labels), num_images).reshape(-1, 1)
    X = g_model.predict([noise, labels])
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0

    os.makedirs(os.path.join(output_dir, f"{epoch}"), exist_ok=True)

    for i in range(num_labels * num_images):
        # Create a plot for the current generated image
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(X[i, :, :, 0], cmap='gray_r')
        ax.axis('off')

        plt.savefig(os.path.join(output_dir, f"{epoch}", f"img_{i}.png"))
        plt.close()

    with open(os.path.join(output_dir, f"{epoch}", "labels.txt"), "w") as f:
        f.write('\n'.join(str(label[0]) for label in labels))


def summarize_performance(epoch, g_model, latent_dim):
    n_samples = 10
    noise = np.random.randn(n_samples, latent_dim)
    labels = np.arange(10).reshape(-1, 1)
    X = g_model.predict([noise, labels])
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0

    fig, axs = plt.subplots(1, 10, figsize=(15, 3))

    for i in range(10):
        axs[i].imshow(X[i, :, :, 0], cmap='gray_r')
        axs[i].set_title(f"Label: {i}")
        axs[i].axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    fid_scores = {}
    dims_list = [64, 192, 768, 2048]
    for dims in dims_list:
        fid_file = f"./fid_acgan_{dims}.txt"
        with open(fid_file, 'r') as f:
            for line in f:
                # print(line)
                epoch_from_file, fid = line.split(
                    "epoch = ")[1].split(", fid = ")
                epoch_from_file, fid = int(
                    epoch_from_file.strip()), float(fid.strip())
                if dims not in fid_scores:
                    fid_scores[dims] = {}
                fid_scores[dims][epoch_from_file] = fid

    os.makedirs(f"./figs", exist_ok=True)
    # Add FID scores for the specific epoch to the plot title
    fid_title = " ".join(
        [f"FID({dims})={fid_scores[dims][epoch]:.2f}" for dims in fid_scores])
    fig.suptitle(f"Epoch={epoch} {fid_title}", fontsize=14)

    # Save plot to file
    filename = f'./figs/{epoch}.png'
    plt.savefig(filename)
    plt.close()

    os.makedirs(f"./models", exist_ok=True)
    # Save the generator model
    filename = f'./models/model_{epoch}.h5'
    g_model.save(filename)


def save_loss_plot(d_losses, g_losses, epochs, filename='loss_plot.png'):
    plt.figure()
    plt.plot(np.arange(epochs), d_losses, label='Discriminator Loss')
    plt.plot(np.arange(epochs), g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def train(g_model, d_model, gan_model, dataset, latent_dim, n_batch, n_epochs):
    start_time = time.time()  # Record the start time

    dataset_size = dataset[0].shape[0]
    bat_per_epo = int(dataset_size / n_batch)

    print(f"Number of epochs: {n_epochs}")
    half_batch = int(n_batch / 2)

    log_dir = f"logs"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    tensorboard_callback.set_model(d_model)
    tensorboard_callback.set_model(gan_model)

    d_losses = []
    g_losses = []

    for epoch in range(n_epochs):
        d_loss_epoch = []
        g_loss_epoch = []
        for i in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(
                dataset, half_batch)
            d_loss_real = d_model.train_on_batch(X_real, [y_real, labels_real])
            [X_fake, labels_fake], y_fake = generate_fake_samples(
                g_model, latent_dim, half_batch)
            d_loss_fake = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            d_loss_epoch.append(d_loss)
            [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(
                [z_input, z_labels], [y_gan, z_labels])
            g_loss_epoch.append(g_loss[0])

        d_losses.append(np.mean(d_loss_epoch))
        g_losses.append(np.mean(g_loss_epoch))
        print(
            f'Epoch {epoch}/{n_epochs} Avg D Loss: {d_losses[-1]:.3f}, Avg G Loss: {g_losses[-1]:.3f}')

        # summarize_performance(epoch, g_model, latent_dim)
        # generate(epoch, g_model, latent_dim, './images')

        # Log losses to TensorBoard
        tensorboard_callback.on_epoch_end(
            epoch, {"d_loss": d_losses[-1], "g_loss": g_losses[-1]})
    tensorboard_callback.on_train_end(None)

   # Save loss plot
    save_loss_plot(d_losses, g_losses, n_epochs)

    end_time = time.time()

    t = int(end_time - start_time)
    print(f"Elapsed time: {t:d} seconds")


def create_gif_from_directory(input_dir, gif_path, duration=2):
    # Get a list of image file names in the input directory
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.png')])

    # Read images from the file names
    images = [Image.open(os.path.join(input_dir, image_file))
              for image_file in image_files]

    # Add epoch number on each image
    for epoch, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Epoch: {epoch}", font=font, fill="white")

    # Save the images as a GIF
    imageio.mimsave(gif_path, images, duration=duration,
                    format='GIF', subrectangles=True)


input_dir = './figs'
output_gif_path = './figs/animation.gif'



def main_loop():
    time_list = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    n_batch = 16
    latent_dim = 100
    n_epochs = 10

    for i in range(10):
        start_time = time.time()

        discriminator = define_discriminator()
        generator = define_generator(latent_dim)
        gan_model = define_gan(generator, discriminator)
        dataset = load_real_samples()
        train(generator, discriminator, gan_model,
              dataset, latent_dim, n_batch, n_epochs)

        end_time = time.time()
        t = int(end_time - start_time)
        time_list.append(t)

    return time_list


def plot_time_list(time_list):
    plt.figure()
    plt.plot(range(1, len(time_list) + 1), time_list)
    plt.xlabel("Run")
    plt.ylabel("Elapsed Time (seconds)")
    plt.title("Elapsed Time vs Run")

    average_time = sum(time_list) / len(time_list)
    plt.axhline(average_time, color='r', linestyle='--',
                label=f'Average Time: {average_time:.2f}s')
    plt.legend()

    plt.grid(True)
    plt.savefig("elapsed_time_plot.png")
    plt.show()


if __name__ == "__main__":
    time_list = main_loop()
    plot_time_list(time_list)

