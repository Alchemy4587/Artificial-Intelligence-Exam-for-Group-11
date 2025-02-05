import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class FashionMNISTModelDevelopment:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def load_preprocessed_data(self):
        """Load preprocessed Fashion-MNIST data"""
        filepath = os.path.join(self.models_dir, 'fashion_mnist_preprocessed.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    def apply_clustering(self, X_train):
        flattened_data = X_train.reshape(X_train.shape[0], -1)
        pca = PCA(n_components=50)
        reduced_data = pca.fit_transform(flattened_data)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=10,
            metric='euclidean',
            cluster_selection_epsilon=0.1
        )
        cluster_labels = clusterer.fit_predict(reduced_data)

        plt.figure(figsize=(10, 8))
        pca_viz = PCA(n_components=2)
        viz_data = pca_viz.fit_transform(reduced_data)

        scatter = plt.scatter(viz_data[:, 0], viz_data[:, 1],
                              c=cluster_labels, cmap='Spectral',
                              alpha=0.5)
        plt.colorbar(scatter)
        plt.title('HDBSCAN Clusters of Fashion-MNIST Data')
        plt.savefig(os.path.join(self.models_dir, 'clusters.png'))
        plt.close()

        return cluster_labels, reduced_data

    def create_cnn_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_test, y_test):
        model = self.create_cnn_model()
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1
        )
        datagen.fit(X_train)

        checkpoint_path = os.path.join(self.models_dir, 'best_model.h5')
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=64),
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
        return model, history

    def analyze_clusters(self, cluster_labels, y_train):
        true_labels = np.argmax(y_train, axis=1)
        plt.figure(figsize=(12, 8))
        confusion = np.zeros((len(np.unique(cluster_labels)), len(self.class_names)))

        for i in range(len(cluster_labels)):
            if cluster_labels[i] != -1:
                confusion[cluster_labels[i]][true_labels[i]] += 1

        sns.heatmap(confusion, xticklabels=self.class_names,
                    yticklabels=range(len(np.unique(cluster_labels))),
                    cmap='YlOrRd')
        plt.title('Cluster-Class Distribution')
        plt.xlabel('True Classes')
        plt.ylabel('Clusters')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'cluster_analysis.png'))
        plt.close()

    def run_model_development(self):
        X_train, y_train, X_test, y_test = self.load_preprocessed_data()
        cluster_labels, reduced_data = self.apply_clustering(X_train)

        clustering_results = {
            'cluster_labels': cluster_labels,
            'reduced_data': reduced_data
        }
        with open(os.path.join(self.models_dir, 'clustering_results.pkl'), 'wb') as f:
            pickle.dump(clustering_results, f)

        model, history = self.train_model(X_train, y_train, X_test, y_test)
        self.analyze_clusters(cluster_labels, y_train)


def main():
    model_dev = FashionMNISTModelDevelopment()
    model_dev.run_model_development()


if __name__ == '__main__':
    main()