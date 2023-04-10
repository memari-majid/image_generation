import time
from collections import defaultdict

import imageio
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch
from PIL import Image

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class AHDD(Dataset):
    def __init__(self, X_file_path, y_file_path, transform=None):
        self.X_data = np.load(X_file_path).reshape(-1,
                                                   28, 28).astype(np.float32)
        self.y_data = np.load(y_file_path)
        self.transform = transform
        self.threshold = 100

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        image = self.X_data[index]
        label = self.y_data[index][0]

        # Apply transformations
        image = np.rot90(image, k=-1, axes=(0, 1))
        image = np.fliplr(image)
        image = np.where(image > self.threshold, 255, 0)

        if self.transform:
            image = Image.fromarray((image).astype(np.uint8))
            image = self.transform(image)

        image = image / 255.0

        return image, label


def load_arabic_dataset(batch_size, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    X_file_path = '/home/siu853655961/image_generation/data/train_images.npy'
    y_file_path = '/home/siu853655961/image_generation/data/train_labels.npy'

    dataset = AHDD(X_file_path, y_file_path, transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28 * 28)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


def generate_images(num_images, vae, device, latent_size, conditional):
    vae.eval()
    z = torch.randn(num_images, latent_size).to(device)

    c = torch.arange(0, 10).long().unsqueeze(1).repeat(
        num_images // 10 + 1, 1).view(-1)[:num_images].to(device)

    generated_images = vae.inference(z, c=c if conditional else None)

    return generated_images, c


# Generate and save images and labels for VAEs
def generate(vae, device, latent_size, conditional, epoch, output_dir):
    num_labels = 10
    num_images = 1000
    vae.eval()
    z = torch.randn(num_labels * num_images, latent_size).to(device)
    c = torch.arange(0, num_labels).unsqueeze(
        1).repeat(1, num_images).view(-1, 1).to(device)

    generated_images = vae.inference(z, c=c if conditional else None)

    os.makedirs(os.path.join(output_dir, f"{epoch}"), exist_ok=True)

    for i in range(num_labels * num_images):
        img_array = generated_images[i].squeeze().cpu().detach().numpy()
        img_array = img_array.reshape(28, 28)  # Reshape the image array

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(img_array, cmap='gray_r')
        ax.axis('off')

        plt.savefig(os.path.join(output_dir, f"{epoch}", f"img_{i}.png"))
        plt.close()

    with open(os.path.join(output_dir, f"{epoch}", "labels.txt"), "w") as f:
        f.write('\n'.join(str(label.item()) for label in c.squeeze()))


def summarize(vae, device, latent_size, conditional, epoch, output_dir):
    num_labels = 10
    vae.eval()
    z = torch.randn(num_labels, latent_size).to(device)
    c = torch.arange(0, 10).long().unsqueeze(1).to(device)

    generated_images = vae.inference(z, c=c if conditional else None)

    os.makedirs(output_dir, exist_ok=True)

    fid_scores = {}
    dims_list = [64, 192, 768, 2048]
    for dims in dims_list:
        fid_file = f"./fid_cvae_{dims}.txt"
        with open(fid_file, 'r') as f:
            for line in f:
                epoch_from_file, fid = line.split(
                    "epoch = ")[1].split(", fid = ")
                epoch_from_file, fid = int(
                    epoch_from_file.strip()), float(fid.strip())
                if dims not in fid_scores:
                    fid_scores[dims] = {}
                fid_scores[dims][epoch_from_file] = fid

    for i in range(num_labels):
        # Create a plot for the current generated image
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        img_array = generated_images[i].squeeze().cpu().detach().numpy()
        img_array = img_array.reshape(28, 28)  # Reshape the image array
        ax.imshow(img_array, cmap='gray_r')
        ax.set_title(f"Label {i}")
        ax.axis('off')

        plt.close()

    # Create a plot for the entire generated images
    fig, axs = plt.subplots(1, num_labels, figsize=(15, 3))

    for i, (label, ax) in enumerate(zip(c, axs.ravel())):
        img_array = generated_images[i].squeeze().cpu().detach().numpy()
        img_array = img_array.reshape(28, 28)  # Reshape the image array
        ax.imshow(img_array, cmap='gray_r')
        ax.set_title(f"Label {label.item()}")
        ax.axis('off')

    # Add a common title to the entire plot
    fid_title = " ".join(
        [f"FID({dims})={fid_scores[dims][epoch]:.2f}" for dims in fid_scores])
    fig.suptitle(f"Epoch={epoch} {fid_title}", fontsize=14)

    # Save the entire plot
    plt.savefig(f"./{output_dir}/{epoch}.png")
    plt.close()


# Add this function to plot the loss per epoch
def plot_loss_per_epoch(logs, output_dir='./'):
    plt.figure()
    plt.plot(logs['epoch_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.show()


def main(seed, epochs, batch_size, learning_rate, encoder_layer_sizes,
         decoder_layer_sizes, latent_size, conditional):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = load_arabic_dataset(batch_size=batch_size)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        conditional=conditional,
        num_labels=10 if conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    logs = defaultdict(list)

    # Create a TensorBoard SummaryWriter
    writer = SummaryWriter('logs')

    for epoch in range(epochs):
        epoch_loss = 0  # Add this line to initialize epoch_loss

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            # Log loss to TensorBoard
            writer.add_scalar('Loss/', loss.item(),
                              epoch * len(data_loader) + iteration)

            if conditional:
                c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                z = torch.randn([c.size(0), latent_size]).to(device)
                x = vae.inference(z, c=c)
            else:
                z = torch.randn([10, latent_size]).to(device)
                x = vae.inference(z)

            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)
        print(f"Epoch {epoch}/{epochs}, Avg Loss {epoch_loss:.2f}")

        logs['epoch_loss'].append(epoch_loss)
        # summarize(vae, device, latent_size, conditional, epoch, './figs')

        # generate(
        #     vae=vae,
        #     device=device,
        #     latent_size=latent_size,
        #     conditional=conditional,
        #     epoch=epoch,
        #     output_dir='./images'
        # )

    plot_loss_per_epoch(logs)
    # Close the TensorBoard SummaryWriter
    writer.close()

# create_gif_from_directory(input_dir, output_gif_path, font_path)
def create_gif_from_directory(input_dir, gif_path, font_path, duration=2):
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


os.makedirs(f"./figs", exist_ok=True)
input_dir = './figs'
output_gif_path = './figs/animation.gif'
font_path = "Roboto-Regular.ttf"

# create_gif_from_directory(input_dir, output_gif_path, font_path)

def main_loop():
    time_list = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for i in range(10):
        start_time = time.time()

        main(
            seed=0,
            epochs=10,
            batch_size=16,
            learning_rate=0.001,
            encoder_layer_sizes=[784, 512],
            decoder_layer_sizes=[512, 784],
            latent_size=10,
            conditional=True
        )

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



