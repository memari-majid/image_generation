import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize


def read_generated_images_and_labels(epoch, path):
    image_file = f"{path}/images/{epoch}/images.npy"
    label_file = f"{path}/images/{epoch}/labels.npy"

    # Load images and labels from numpy files
    images = np.load(image_file)
    labels = np.load(label_file)
    labels = np.squeeze(labels)

    # Normalize the images
    images = images.astype("float32") / 255.0
    images = np.reshape(images, (-1, 28, 28, 1))

    return images, labels


def create_model(model_name):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1), kernel_initializer="he_normal",
                name=f"conv2d_{model_name}"
            ),
            tf.keras.layers.BatchNormalization(
                name=f"batch_norm_{model_name}"),
            tf.keras.layers.MaxPooling2D(
                (2, 2), name=f"max_pooling2d_{model_name}"),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", kernel_initializer="he_normal", name=f"conv2d_1_{model_name}"
            ),
            tf.keras.layers.BatchNormalization(
                name=f"batch_norm_1_{model_name}"),
            tf.keras.layers.MaxPooling2D(
                (2, 2), name=f"max_pooling2d_1_{model_name}"),
            tf.keras.layers.Flatten(name=f"flatten_{model_name}"),
            tf.keras.layers.Dense(
                128, activation="relu", kernel_initializer="he_normal", name=f"dense_{model_name}"),
            tf.keras.layers.Dropout(0.5, name=f"dropout_{model_name}"),
            tf.keras.layers.Dense(
                10, activation="softmax", kernel_initializer="he_normal", name=f"dense_1_{model_name}"),
        ]
    )
    optim = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_and_evaluate_model(model_name, X_train, Y_train, X_test, Y_test, epochs):
    # One-hot encode the labels
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)

    model = create_model(model_name)

    # The rest of the function remains unchanged...

    # One-hot encode the labels
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Create a model for Arabic handwritten digit recognition
    model = create_model(model_name)

    # Define early stopping based on validation accuracy
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=2,
                                   mode='max', baseline=0.9)

    # Save the best weights based on validation accuracy
    checkpoint_filepath = f"{model_name}_best_weights.h5"
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
                                       mode='max', save_best_only=True, save_weights_only=True)

    # Train the model using the combined data
    model.fit(X_train, Y_train, epochs=epochs,
              batch_size=64, validation_data=(X_test, Y_test), verbose=0,
              callbacks=[early_stopping, model_checkpoint])

    # Load the best weights and evaluate the model
    model.load_weights(checkpoint_filepath)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    return model, test_acc


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    confusion_mat = confusion_matrix(y_true, y_pred)
    return precision, recall, f1, confusion_mat


def read_fid_scores(dimension, epochs):
    fid_scores = {"acgan": {}, "cvae": {}}

    for dim in dimension:
        for acgan_type in ["acgan", "cvae"]:
            file = f"{acgan_type}/fid_{acgan_type}_{dim}.txt"
            with open(file, 'r') as f:
                lines = f.readlines()
                scores = []
                for line in lines:
                    splitted_line = line.strip().split(", ")
                    epoch, score = int(splitted_line[0].split(" = ")[1]), float(
                        splitted_line[1].split(" = ")[1])
                    scores.append((epoch, score))
                fid_scores[acgan_type][dim] = scores
    return fid_scores

# Save FID scores plot
def plot_fid_scores(fid_scores, dimensions, epochs):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for idx, dim in enumerate(dimensions):
        row = idx // 2
        col = idx % 2
        for gan_type in ["acgan", "cvae"]:
            epoch_fid = np.array(fid_scores[gan_type][dim])
            axes[row, col].plot(epoch_fid[:, 0], epoch_fid[:, 1], label=gan_type)
            axes[row, col].set_title(f'FID Scores for {dim}-dimension')
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].set_ylabel("FID Score")
            axes[row, col].legend()
            axes[row, col].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fid_scores.png'))
    plt.close()



def plot_test_accuracy(epochs_list, acgan_test_acc_list, cvae_test_acc_list):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_list, acgan_test_acc_list, label="ACGAN Test Accuracy")
    ax.plot(epochs_list, cvae_test_acc_list, label="CVAE Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.set_title("Test Accuracy vs. Epoch for ACGAN and CVAE")
    plt.savefig(os.path.join(output_dir, 'test_accuracy_vs_epoch.png'))
    plt.close()

    

def visualize_saliency_map(model, X_test, idx, output_dir):
    image = X_test[idx]
    preprocessed_image = image / 255.0

    def loss(output):
        return output

    gradcam = Gradcam(model)
    saliency_map = gradcam(loss, np.array([preprocessed_image]))
    saliency_map = normalize(saliency_map)

    plt.imshow(image)
    plt.imshow(saliency_map[0], cmap='jet', alpha=0.5)
    plt.axis('off')

    output_path = os.path.join(output_dir, f'saliency_map_{idx}.png')
    plt.savefig(output_path)
    plt.close()


# Set GPU device to 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Create an output directory if it doesn't exist
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

real_train_images = np.load("data/train_images.npy")
real_train_labels = np.load("data/train_labels.npy")

# Select only the first 1000 real train images and labels
real_train_images = real_train_images[:500]
real_train_labels = real_train_labels[:500]

real_test_images = np.load(f"data/test_images.npy")
real_test_labels = np.load(f"data/test_labels.npy")

# Reshape the images to have a 3D tensor shape (samples, height, width, channels)
real_train_images = np.reshape(real_train_images, (-1, 28, 28, 1))
real_test_images = np.reshape(real_test_images, (-1, 28, 28, 1))

# Shuffle the real data
real_train_images, real_train_labels = shuffle(real_train_images, real_train_labels, random_state=42)

# Preprocess real images and labels
real_train_images = real_train_images.astype("float32") / 255.0  # Normalize the images
X_train = np.array(real_train_images).reshape(-1, 28, 28, 1)
Y_train = np.array(real_train_labels).astype(int).reshape(-1, 1)

# Preprocess real test images and labels
real_test_images = real_test_images.astype("float32") / 255.0  # Normalize the images
X_test = np.array(real_test_images).reshape(-1, 28, 28, 1)
Y_test = np.array(real_test_labels).astype(int).reshape(-1, 1)

Y_train = np.squeeze(Y_train)
Y_test = np.squeeze(Y_test)

# Train and evaluate the model on real images and labels
model_real, real_test_acc = train_and_evaluate_model(
    'real', X_train, Y_train, real_test_images, real_test_labels, epochs=10)




# Initialize lists to store metrics
acgan_test_acc_list = []
cvae_test_acc_list = []
real_precision_list = []
real_recall_list = []
real_f1_list = []
acgan_precision_list = []
acgan_recall_list = []
acgan_f1_list = []
cvae_precision_list = []
cvae_recall_list = []
cvae_f1_list = []
epochs_list = []

for epoch in range(10):
    # Read generated images and labels
    acgan_images, acgan_labels = read_generated_images_and_labels(epoch, "acgan")
    cvae_images, cvae_labels = read_generated_images_and_labels(epoch, "cvae")

    # read acgan generated images and labels
    X_train_acgan = acgan_images

    # Reshape Y_train and acgan_labels to have the same dimensions (1D)
    Y_train_reshaped = Y_train.reshape(-1)
    acgan_labels_reshaped = np.array(acgan_labels).astype(int).reshape(-1)

    Y_train_acgan = acgan_labels_reshaped

    # read cvae generated images and labels
    X_train_cvae = cvae_images

    # Reshape Y_train and cvae_labels to have the same dimensions (1D)
    Y_train_reshaped = Y_train.reshape(-1)
    cvae_labels_reshaped = np.array(cvae_labels).astype(int).reshape(-1)

    # Concatenate real data with CVAE data
    X_train_combined =  X_train_cvae
    Y_train_combined = cvae_labels_reshaped

    # Shuffle the combined data (CVAE)
    X_train_cvae, Y_train_cvae = shuffle(X_train_combined, Y_train_combined, random_state=42)


    # Train and evaluate models on the combined data
    model_acgan, acgan_test_acc = train_and_evaluate_model(
        'acgan', X_train_acgan, Y_train_acgan, X_test, Y_test, epochs=10)
    model_cvae, cvae_test_acc = train_and_evaluate_model(
        'cvae', X_train_cvae, Y_train_cvae, X_test, Y_test, epochs=10)
    
    # Append the metrics to the lists
    acgan_test_acc_list.append(acgan_test_acc)
    cvae_test_acc_list.append(cvae_test_acc)
    epochs_list.append(epoch)


    # Calculate metrics for real data model
    Y_real_pred = model_real.predict(X_test).argmax(axis=1)
    real_precision, real_recall, real_f1, real_confusion_mat = calculate_metrics(Y_test, Y_real_pred)

    # Calculate metrics for ACGAN combined data model
    Y_acgan_pred = model_acgan.predict(X_test).argmax(axis=1)
    acgan_precision, acgan_recall, acgan_f1, acgan_confusion_mat = calculate_metrics(Y_test, Y_acgan_pred)

    # Calculate metrics for CVAE combined data model
    Y_cvae_pred = model_cvae.predict(X_test).argmax(axis=1)
    cvae_precision, cvae_recall, cvae_f1, cvae_confusion_mat = calculate_metrics(Y_test, Y_cvae_pred)


    # Append the metrics to the lists
    real_precision_list.append(real_precision)
    real_recall_list.append(real_recall)
    real_f1_list.append(real_f1)
    acgan_precision_list.append(acgan_precision)
    acgan_recall_list.append(acgan_recall)
    acgan_f1_list.append(acgan_f1)
    cvae_precision_list.append(cvae_precision)
    cvae_recall_list.append(cvae_recall)
    cvae_f1_list.append(cvae_f1)
    epochs_list.append(epoch)

    # Dimensions and epochs
    dimension = [64, 192, 768, 2048]

    # Read and plot FID scores
    fid_scores = read_fid_scores(dimension, epoch)
    plot_fid_scores(fid_scores, dimension, epoch)

    # # Save and show the test accuracy plot
    # plt.plot(epochs_list, acgan_test_acc_list, label="ACGAN Test Accuracy")
    # plt.plot(epochs_list, cvae_test_acc_list, label="CVAE Test Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Test Accuracy (%)")
    # plt.legend()
    # plt.title("Test Accuracy vs. Epoch for ACGAN and CVAE")
    # plt.savefig(os.path.join(output_dir, 'test_accuracy_vs_epoch.png'))
    # plt.show()

        # Visualize sal
    # Visualize saliency maps for a few test images
    # Replace these with the indices of the images you want to visualize
    test_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    output_dir_acgan = f'./saliency_maps_acgan/epoch_{epoch}'
    output_dir_cvae = f'./saliency_maps_cvae/epoch_{epoch}'

    if not os.path.exists(output_dir_acgan):
        os.makedirs(output_dir_acgan)
    if not os.path.exists(output_dir_cvae):
        os.makedirs(output_dir_cvae)

    layer_name_acgan = "conv2d_1_acgan"
    layer_name_cvae = "conv2d_1_cvae"

    layer_acgan = model_acgan.get_layer(layer_name_acgan)
    layer_cvae = model_cvae.get_layer(layer_name_cvae)

    # Replace the activation function of the layer with a linear activation
    layer_acgan.activation = tf.keras.activations.linear
    layer_cvae.activation = tf.keras.activations.linear

    # Re-build the models
    model_acgan = tf.keras.models.clone_model(model_acgan)
    model_cvae = tf.keras.models.clone_model(model_cvae)

    # Compile the models to apply the modifications
    model_acgan.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model_cvae.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
        # Visualize the saliency maps
    for idx in test_indices:
        visualize_saliency_map(model_acgan, X_test, idx, output_dir_acgan)
        visualize_saliency_map(model_cvae, X_test, idx, output_dir_cvae)

# Call the function with your data
plot_test_accuracy(epochs_list, acgan_test_acc_list, cvae_test_acc_list)

# Plot the metrics
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

axes[0, 0].plot(epochs_list, acgan_test_acc_list, label="ACGAN Data")
axes[0, 0].plot(epochs_list, cvae_test_acc_list, label="CVAE Data")
axes[0, 0].set_title("Test Accuracy")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid()

axes[0, 1].plot(epochs_list, real_precision, label="Real Data")
axes[0, 1].plot(epochs_list, acgan_precision, label="ACGAN Data")
axes[0, 1].plot(epochs_list, cvae_precision, label="CVAE Data")
axes[0, 1].set_title("Precision")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].legend()
axes[0, 1].grid()

axes[1, 0].plot(epochs_list, real_recall, label="Real Data")
axes[1, 0].plot(epochs_list, acgan_recall, label="ACGAN Data")
axes[1, 0].plot(epochs_list, cvae_recall, label="CVAE Data")
axes[1, 0].set_title("Recall")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Recall")
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].plot(epochs_list, real_f1, label="Real Data")
axes[1, 1].plot(epochs_list, acgan_f1, label="ACGAN Data")
axes[1, 1].plot(epochs_list, cvae_f1, label="CVAE Data")
axes[1, 1].set_title("F1 Score")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.savefig("model_metrics.png")
plt.show()