# 🎯 CIFAR-10 Object Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**, which consists of 10 categories of 32x32 color images:

* ✈️ Airplane
* 🚗 Automobile
* 🐦 Bird
* 🐱 Cat
* 🦌 Deer
* 🐶 Dog
* 🐸 Frog
* 🐴 Horse
* 🚢 Ship
* 🚚 Truck

The objective is to automatically detect and classify objects using deep learning techniques.

---

## 📁 Project Structure

```
ObjectRecognitionImages/
│
├── cifar10_cnn.py         # Main Python script
├── setup_and_run.bat      # Windows setup and execution script
├── setup_and_run.sh       # macOS/Linux setup script
└── README.md              # Documentation
```

---

## ⚙️ Setup Instructions

### ✅ Requirements

Make sure Python and pip are installed. Then install the required libraries using:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Alternatively, you can use the provided setup scripts:

* **Windows**: `setup_and_run.bat` *(may take some time to complete)*
* **macOS/Linux**: `setup_and_run.sh`

---

## 🚀 How to Run

### 💻 On Windows:

1. Download or clone the repository.
2. Run `setup_and_run.bat` (or run `cifar10_cnn.py` manually after installing dependencies).

### 💻 On macOS/Linux:

1. Open a terminal in the project directory.
2. Run the following:

   ```bash
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

> The script will set up a virtual environment, install the dependencies, and start training the model.

---

## 📊 Output After Training

Once the model finishes training, it will:

* Display **Test Accuracy**
* Show **Precision, Recall, and F1-Score**
* Plot a **Confusion Matrix** using Seaborn heatmap

---

## 🧠 Model Summary

The CNN architecture used includes:

* 3 × Convolutional + ReLU layers
* 2 × Max Pooling layers
* 1 × Flatten + Dropout + Dense layer
* 1 × Output layer with Softmax (10 classes)

---

## 📚 References

* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [Kaggle: CIFAR-10 CNN for Beginners](https://www.kaggle.com/code/roblexnana/cifar10-with-cnn-for-beginer)
