import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


class FashionMNISTProcessor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def load_dataset(self):
        """Load Fashion-MNIST dataset from TensorFlow"""
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test

    def explore_dataset(self, X_train, y_train):
        """Perform dataset exploration and visualization"""
        # Dataset information
        dataset_info = {
            'n_train_samples': len(X_train),
            'image_shape': X_train[0].shape,
            'n_classes': len(self.class_names),
            'samples_per_class': np.bincount(y_train)
        }
        print("\nDataset Exploration:")
        print(f"Training samples: {dataset_info['n_train_samples']}")
        print(f"Image shape: {dataset_info['image_shape']}")
        print(f"Number of classes: {dataset_info['n_classes']}")

        # Visualize sample images
        plt.figure(figsize=(15, 3))
        random_indices = np.random.choice(len(X_train), 10, replace=False)
        for i, idx in enumerate(random_indices):
            plt.subplot(1, 10, i + 1)
            plt.imshow(X_train[idx], cmap='gray')
            plt.title(self.class_names[y_train[idx]])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.models_dir, 'sample_images.png'))
        plt.close()

        # Class distribution
        dist_df = pd.DataFrame({
            'Class': self.class_names,
            'Number of Images': dataset_info['samples_per_class']
        }).sort_values('Number of Images', ascending=False)

        plt.figure(figsize=(12, 5))
        sns.barplot(x='Class', y='Number of Images', data=dist_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Images per Class')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.models_dir, 'class_distribution.png'))
        plt.close()

        return dataset_info

    def preprocess_data(self, X_train, y_train, X_test, y_test):
        """Preprocess image data and labels"""
        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        return X_train, y_train, X_test, y_test

    def save_preprocessed_data(self, X_train, y_train, X_test, y_test, dataset_info):
        """Save preprocessed data and metadata"""
        preprocessed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'dataset_info': dataset_info,
            'class_names': self.class_names
        }

        filepath = os.path.join(self.models_dir, 'fashion_mnist_preprocessed.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessed_data, f)

        print(f"\nPreprocessed data saved to {filepath}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

    def process_dataset(self):
        """Execute full data processing pipeline"""
        # Load original dataset
        X_train, y_train, X_test, y_test = self.load_dataset()

        # Explore dataset and generate visualizations
        dataset_info = self.explore_dataset(X_train, y_train)

        # Preprocess data
        X_train, y_train, X_test, y_test = self.preprocess_data(X_train, y_train, X_test, y_test)

        # Save preprocessed data
        self.save_preprocessed_data(X_train, y_train, X_test, y_test, dataset_info)


def main():
    processor = FashionMNISTProcessor()
    processor.process_dataset()


if __name__ == '__main__':
    main()