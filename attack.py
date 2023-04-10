import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score



def fgsm_attack(image, epsilon, model, true_label):
    # Cast the image to a tensor
    image = tf.cast(image, tf.float32)

    # Get the gradient of the loss with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(image)
        logits = model(image)
        true_label = tf.reshape(true_label, logits.shape)
        loss = tf.keras.losses.categorical_crossentropy(true_label, logits)
    gradient = tape.gradient(loss, image)

    # Get the sign of the gradient
    sign_gradient = tf.sign(gradient)

    # Create the adversarial image
    adversarial_image = image + epsilon * sign_gradient
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    # Make a prediction on the original and adversarial images
    orig_logits = model(image)
    orig_prediction = tf.argmax(orig_logits, axis=1)
    adv_logits = model(adversarial_image)
    adv_prediction = tf.argmax(adv_logits, axis=1)

    # Compute the accuracy and return the adversarial example, original image and their labels
    acc = accuracy_score(tf.argmax(true_label, axis=1), adv_prediction)
    return adversarial_image.numpy(), acc, image.numpy(), orig_prediction.numpy(), true_label.numpy()




def read_images_and_labels(image_path, label_path):
    images = []
    labels = []

    with open(label_path, 'r') as f:
        label_lines = f.readlines()[:]

    for i, img_name in enumerate(sorted(os.listdir(image_path))):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)

            images.append(img)

            label = int(label_lines[i].strip())
            label_tensor = tf.one_hot(label, depth=10)
            labels.append(label_tensor)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


real_image_path = './real'
real_label_path = './real/labels.txt'

real_images, real_labels = read_images_and_labels(
    real_image_path, real_label_path)

model_type_list = ['acgan', 'cvae']
epochs = range(10)
models = {}

for model_type in model_type_list:
    for epoch in epochs:
        model_name = f'{model_type}_{epoch}.h5'
        model_path = os.path.join('./models/', model_name)
        model = load_model(model_path)
        models[model_name] = model


epsilon = 0.01

for model_name, model in models.items():
    print(f"Testing model: {model_name}")

    for i in range(len(real_images)):
        sample_image = real_images[i]
        sample_label = real_labels[i]

        # Reshape the image to be compatible with the model input shape
        sample_image = np.expand_dims(sample_image, axis=0)

        # Create the adversarial example
        adversarial_example = fgsm_attack(
            sample_image, epsilon, model, sample_label)

        # Test the model's performance on the adversarial example
        original_prediction = model.predict(sample_image)
        adversarial_prediction = model.predict(adversarial_example)

        print(f"Image {i + 1}:")
        print("Original prediction: ", np.argmax(original_prediction))
        print("Adversarial prediction: ", np.argmax(adversarial_prediction))
        print("\n")


epsilon_values = np.linspace(0, 0.1, 11)
accuracies = {model_name: [] for model_name in models.keys()}


def test_model_on_adversarial_examples(model, epsilon, real_images, real_labels):
    correct_predictions = 0
    total_images = len(real_images)

    for i in range(total_images):
        sample_image = real_images[i]
        sample_label = real_labels[i]
        sample_image = np.expand_dims(sample_image, axis=0)
        adversarial_example = fgsm_attack(sample_image, epsilon, model, sample_label)
        adversarial_prediction = model.predict(adversarial_example)

        if np.argmax(adversarial_prediction) == sample_label:
            correct_predictions += 1

    return correct_predictions / total_images


for model_name, model in models.items():
    print(f"Testing model: {model_name}")

    for epsilon in epsilon_values:
        accuracy = test_model_on_adversarial_examples(
            model, epsilon, real_images, real_labels)
        accuracies[model_name].append(accuracy)



epsilon = 0.01

for model_name, model in models.items():
    print(f"Testing model: {model_name}")

    for i in range(len(real_images)):
        sample_image = real_images[i]
        sample_label = real_labels[i]

        # Reshape the image to be compatible with the model input shape
        sample_image = np.expand_dims(sample_image, axis=0)

        # Create the adversarial example
        adversarial_example, acc, orig_image, orig_prediction, true_label = fgsm_attack(
            sample_image, epsilon, model, sample_label)

        # Print the original and adversarial images and labels
        print(f"Image {i + 1}:")
        print("Original label: ", np.argmax(true_label))
        print("Original prediction: ", orig_prediction[0])
        plt.imshow(orig_image[0, :, :, 0], cmap='gray')
        plt.show()
        print("Adversarial label: ", np.argmax(true_label))
        print("Adversarial prediction: ", np.argmax(model.predict(adversarial_example)))
        plt.imshow(adversarial_example[0, :, :, 0], cmap='gray')
        plt.show()
        print("\n")



# Create a directory to save the plots if it doesn't exist
if not os.path.exists("./attack"):
    os.makedirs("./attack")

# Plot the results
plt.figure(figsize=(10, 6))

for model_name, accuracy_values in accuracies.items():
    plt.plot(epsilon_values, accuracy_values, label=model_name)

plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Model robustness to different levels of noise")
plt.legend()

# Save the plot in the attack directory
plt.savefig("./attack/model_robustness_comparison.png")

# Display the plot
plt.show()