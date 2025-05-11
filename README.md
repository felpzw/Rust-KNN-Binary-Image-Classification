# **KNN-Rust-Binary-Image-Classifier**

A simple binary image classifier implemented in Rust using the K-Nearest Neighbors (KNN) algorithm. This project allows you to train a model to classify images into two categories (e.g., "GOOD" and "BAD") and then use the trained model to classify new images.

---

## **Features**
- Preprocesses images by converting them to grayscale, resizing, and normalizing pixel values.
- Trains a KNN model using labeled image datasets.
- Saves and loads the trained model for future use.
- Classifies new images based on the trained model.
- Provides detailed logging for debugging and monitoring.

---

## **Requirements**

### **Dependencies**
- Rust (latest stable version)
- Required crates:
    - `clap` (for command-line argument parsing)
    - `image` (for image processing)
    - `ndarray` (for numerical arrays)
    - `bincode` (for model serialization)
    - `smartcore` (for KNN implementation)
    - `env_logger` (for logging)

To install the dependencies, add the following to your `Cargo.toml`:
```toml
[dependencies]
clap = { version = "4.0", features = ["derive"] }
image = "0.24"
ndarray = "0.15"
bincode = "1.3"
smartcore = "0.3"
env_logger = "0.10"
```

---

## **How to Use**

### **1. Train the Model**
To train the model, you need two directories:
- One containing images labeled as "GOOD".
- Another containing images labeled as "BAD".

Run the following command:
```bash
cargo run -- train --good-images <path_to_good_images> --bad-images <path_to_bad_images> --output <output_model_path> --width <image_width> --height <image_height> --k <number_of_neighbors>
```

#### **Arguments**
- `--good-images`: Path to the directory containing "GOOD" images.
- `--bad-images`: Path to the directory containing "BAD" images.
- `--output`: Path to save the trained model (default: `model.bin`).
- `--width`: Width to resize the images (default: `64`).
- `--height`: Height to resize the images (default: `64`).
- `--k`: Number of neighbors for the KNN algorithm (default: `3`).

#### **Example**
```bash
cargo run -- train --good-images ./data/good --bad-images ./data/bad --output model.bin --width 64 --height 64 --k 3
```

---

### **2. Classify an Image**
Once the model is trained, you can classify new images using the `classify` command.

Run the following command:
```bash
cargo run -- classify --model <path_to_trained_model> --image <path_to_image>
```

#### **Arguments**
- `--model`: Path to the trained model file.
- `--image`: Path to the image to classify.

#### **Example**
```bash
cargo run -- classify --model model.bin --image ./data/test_image.png
```

#### **Output**
The program will output the classification result, e.g.:
```
Classification result: GOOD
```

---

## **Project Structure**
- `src/main.rs`: Entry point of the application. Handles command-line arguments and delegates tasks to the appropriate module.
- `src/train.rs`: Contains the logic for training the KNN model.
- `src/predict.rs`: Contains the logic for classifying images using the trained model.
- `src/model.rs`: Defines the `ImageClassifier` struct and methods for saving/loading the model.

---

## **How It Works**

### **Training**
1. Images are preprocessed (grayscale conversion, resizing, normalization).
2. Features are extracted and labeled as "GOOD" or "BAD".
3. A KNN model is trained using the labeled features and saved to a file.

### **Classification**
1. A new image is preprocessed in the same way as during training.
2. The KNN model predicts the class of the image based on its nearest neighbors in the training data.

---

## **Example Dataset**
You can create a dataset with the following structure:
```
data/
├── good/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── bad/
│   ├── image1.png
│   ├── image2.png
│   └── ...
```

---

## **Logging**
The application uses `env_logger` for logging. To enable debug logs, set the `RUST_LOG` environment variable:
```bash
RUST_LOG=debug cargo run -- <command>
```

---

## **Future Improvements**
- Add confidence scores to the classification output.
- Support for multi-class classification.
- Implement additional distance metrics for KNN.
- Add unit tests for all modules.

---

## **License**
This project is licensed under the MIT License. Feel free to use and modify it as needed.
