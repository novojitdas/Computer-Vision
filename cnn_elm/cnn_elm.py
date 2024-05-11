import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from tensorflow.keras.applications import ResNet101, DenseNet169, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy

# Simple ELM Classifier Implementation
class ELMClassifierFromScratch:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Random weights and biases for input to hidden layer
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.rand(1, self.hidden_size)

        # Placeholder for output weights
        self.weights_hidden_output = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _one_hot_encode(self, labels):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

    def _add_bias(self, data):
        return np.hstack([data, np.ones((data.shape[0], 1))])

    def fit(self, X, y):
        X = self._add_bias(X)

        # Calculate hidden layer output
        hidden_output = self._sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)

        # Calculate output layer weights using pseudoinverse
        self.weights_hidden_output = np.dot(np.linalg.pinv(hidden_output), y)

    def predict(self, X):
        X = self._add_bias(X)

        # Calculate hidden layer output
        hidden_output = self._sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)

        # Calculate final output
        output = np.dot(hidden_output, self.weights_hidden_output)
        return output

    def score(self, X, y):
        predictions = self.predict(X)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

# Function to build CNN-ELM model
def build_cnn_elm_model(hidden_size):
    # Define input shape
    input_shape = (128, 128, 3)

    # Define base CNN models
    base_cnn_resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
    base_cnn_resnet101.trainable = False

    base_cnn_densenet169 = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    base_cnn_densenet169.trainable = False

    base_cnn_resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_cnn_resnet50.trainable = False

    # Extract features using the base CNN models
    input_layer = Input(shape=input_shape)

    resnet101_features = base_cnn_resnet101(input_layer)
    resnet101_features = GlobalAveragePooling2D()(resnet101_features)

    densenet169_features = base_cnn_densenet169(input_layer)
    densenet169_features = GlobalAveragePooling2D()(densenet169_features)

    resnet50_features = base_cnn_resnet50(input_layer)
    resnet50_features = GlobalAveragePooling2D()(resnet50_features)

    # Concatenate the extracted features
    concatenated_features = Concatenate()([resnet101_features, densenet169_features, resnet50_features])

    # Dense layer for dimensionality reduction
    dense_layer = Dense(hidden_size, activation='relu')(concatenated_features)

    # Create an instance of ELMClassifierFromScratch
    elm_model = ELMClassifierFromScratch(hidden_size, 64, num_classes)

    # Create a model
    model = Model(inputs=input_layer, outputs=dense_layer)

    return model, elm_model


# Define paths to your dataset
train_data_dir = './data/WhichWaste_dataset/train'
validation_data_dir = './data/WhichWaste_dataset/val'

# Set image size and batch size
img_size = (128, 128)
batch_size = 32

# Create ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and prepare validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Display class indices
print("Class Indices:", train_generator.class_indices)

# Assuming y_train_encoded and y_test_encoded are one-hot encoded labels
num_classes = len(train_generator.class_indices)
y_train_encoded = to_categorical(train_generator.classes, num_classes=num_classes)
y_test_encoded = to_categorical(validation_generator.classes, num_classes=num_classes)

# Print number of classes
print("Number of classes:", num_classes)

# Build CNN-ELM models
hidden_size = 64  # Adjust as needed
model_cnn_elm, elm_model = build_cnn_elm_model(hidden_size)

# Modify the output layer to match the number of classes
output_layer = Dense(num_classes, activation='softmax')(model_cnn_elm.layers[-1].output)

# Create a new model with the modified output layer
model_cnn_elm = Model(inputs=model_cnn_elm.input, outputs=output_layer)

# Train the CNN part of the model
epochs = 1  # Adjust as needed
model_cnn_elm.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
model_cnn_elm.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Extract features from the trained CNN
X_train_features = model_cnn_elm.predict(train_generator)
X_test_features = model_cnn_elm.predict(validation_generator)

# Train the ELM layer
elm_model.fit(X_train_features, y_train_encoded)

# Evaluate the ELM layer
accuracy = elm_model.score(X_test_features, y_test_encoded)
print(f'Accuracy: {accuracy}')
