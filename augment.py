import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load your dataset
# Example: 
# images, labels = load_your_dataset()  # Replace with your dataset loading logic

# Example dimensions
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256  # Set according to your dataset

# Initialize the ImageDataGenerator with subtle augmentation parameters
data_gen = ImageDataGenerator(
    rotation_range=10,             # Randomly rotate images in the range (degrees)
    width_shift_range=0.1,         # Randomly translate images horizontally (10% of width)
    height_shift_range=0.1,        # Randomly translate images vertically (10% of height)
    shear_range=0.1,               # Shear intensity (10% of max shearing)
    zoom_range=0.1,                # Randomly zoom into images (up to 10%)
    horizontal_flip=True,          # Randomly flip images
    fill_mode='nearest',           # Fill in missing pixels after transformations
    brightness_range=[0.9, 1.1],   # Randomly change brightness (10% variation)
    channel_shift_range=10.0       # Slightly change color channels (e.g., for green intensity)
)

# Example function to visualize augmented images
def visualize_augmentation(image, augmentations, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        augmented_image = augmentations(image)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(augmented_image.astype('uint8'))  # Ensure proper dtype for displaying
        plt.axis('off')
    plt.show()

# Example: Load an image for augmentation demonstration
# Replace 'your_image_path' with the path to one of your leaf images
image = tf.keras.preprocessing.image.load_img('your_image_path', target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
image = tf.keras.preprocessing.image.img_to_array(image)  # Convert to array
image = tf.expand_dims(image, axis=0)  # Add batch dimension

# Create an iterator for generating augmented images
augmented_iterator = data_gen.flow(image)

# Visualize augmented images
visualize_augmentation(image[0], lambda img: next(augmented_iterator)[0])

# To use in model training
# model.fit(data_gen.flow(training_images, training_labels, batch_size=32), epochs=50)
