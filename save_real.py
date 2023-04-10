import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def read_csv_data(images_csv, labels_csv):
    images_df = pd.read_csv(images_csv, header=None)
    labels_df = pd.read_csv(labels_csv, header=None)
    
    images = images_df.to_numpy(dtype=np.float32)
    labels = labels_df.to_numpy(dtype=int)
    
    return images, labels

def save_images_and_labels(images, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = images.shape[0]
    for i in range(num_images):
        img_array = images[i].reshape(28, 28)
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(img_array, cmap='gray_r')
        ax.axis('off')

        plt.savefig(os.path.join(output_dir, f"img_{i}.png"))
        plt.close()

    with open(os.path.join(output_dir, f"labels.txt"), "w") as f:
        f.write('\n'.join(str(label[0]) for label in labels))

def main():
    # Specify the input CSV files and output directory
    train_images_csv = "/home/siu853655961/image_generation/data/trainImages.csv"
    train_labels_csv = "/home/siu853655961/image_generation/data/trainLabel.csv"
    output_dir = "./real"

    # Read train images and labels from CSV files
    train_images, train_labels = read_csv_data(train_images_csv, train_labels_csv)


    # Save train images and labels as PNG files and text files
    save_images_and_labels(train_images, train_labels, output_dir)


if __name__ == "__main__":
    main()
