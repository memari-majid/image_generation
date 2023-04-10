# from fid import calculate_fid_given_path, calculate_frechet_distance
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from tensorboardX import SummaryWriter
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3
import pandas as pd


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    model.eval()
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def calculate_fid_given_path(path, batch_size, device, dims, num_workers=1):
    """Calculates the FID of one path"""
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = compute_statistics_of_path(
        path, model, batch_size, dims, device, num_workers)
    return m1, s1


def calculate_fid_and_plot(gan_type, dims_list, real_path, batch_size, device, num_workers):
    epochs_fid_values = []

    for dims in dims_list:
        writer = SummaryWriter(f'runs/{gan_type}_{dims}')
        os.makedirs(os.path.dirname(
            f'./fid/{gan_type}_{dims}.txt'), exist_ok=True)
        fid_scores_file = f'./fid/fid_{gan_type}_{dims}.txt'

        with open(fid_scores_file, "w") as f:
            m1, s1 = calculate_fid_given_path(
                real_path, batch_size, device, dims, num_workers=num_workers)

            epochs = []
            fid_values = []

            for epoch in range(10):
                images_path = os.path.join(gan_type, "images", str(epoch))
                m2, s2 = calculate_fid_given_path(
                    images_path, batch_size, device, dims, num_workers=num_workers)
                fid_value = calculate_frechet_distance(m1, s1, m2, s2)

                # Log to TensorBoard
                writer.add_scalar('./fid/logs', fid_value, epoch)

                print(f"epoch = {epoch}, fid = {fid_value}")
                f.write(f"epoch = {epoch}, fid = {fid_value:.2f}\n")

                # Save epoch and FID value for plotting
                epochs.append(epoch)
                fid_values.append(fid_value)

            epochs_fid_values.append((epochs, fid_values))
            writer.close()

    return epochs_fid_values


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 512
num_workers = 40
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dims_list = [64, 192, 768, 2048]

real_path = '/home/siu853655961/image_generation/real'

# Calculate FID values for each GAN type and dimensions
gan_fid_values = {}
for gan_type in ['acgan', 'cvae']:
    epochs_fid_values = calculate_fid_and_plot(
        gan_type, dims_list, real_path, batch_size, device, num_workers)
    gan_fid_values[gan_type] = epochs_fid_values

# Plot FID values for ACGAN and CVAE separately for each dimension
for dims_idx, dims in enumerate(dims_list):
    plt.figure()
    for gan_type in ['acgan', 'cvae']:
        epochs, fid_values = gan_fid_values[gan_type][dims_idx]
        plt.plot(epochs, fid_values, label=f'{gan_type}')

    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.title(f"FID Score vs Epoch (dims={dims})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./fid/fid_scores_plot_dims_{dims}.png")
    plt.show()
