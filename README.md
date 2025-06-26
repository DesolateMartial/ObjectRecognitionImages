# CIFAR-10 Object Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**, which includes 10 classes of 32x32 color images:

* Airplane ✈️
* Automobile 🚗
* Bird 🐦
* Cat 🐱
* Deer 🦌
* Dog 🐶
* Frog 🐸
* Horse 🐴
* Ship 🚢
* Truck 🚚

The goal is to automatically detect and recognize objects using image classification techniques.

---

## 📁 Folder Structure

```
ObjectRecognitionImages/
│
├── cifar10_cnn.py         # Main Python script
├── setup_and_run.bat      # For Windows setup and execution
├── setup_and_run.sh       # For macOS/Linux setup (optional)
└── README.md              # Project documentation
```

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

Install the required Python libraries:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Or run the provided setup script:

* **Windows**: `setup_and_run.bat`("wait for sometime....takes lot of time")
* **Mac/Linux**: `setup_and_run.sh`

---

## 🚀 How to Run the Project

### 🖥️ Windows:

1. Clone or download this repository.
2. Double-click `setup_and_run.bat`.(it takes time)

or 
1.Run .py file after installing all libraries.

### 💻 macOS/Linux:

1. Open a terminal in the project folder.
2. Run:

   ```bash
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

> The script will create a virtual environment, install required packages, and execute the model.

---

## 📊 After Training

After training is complete, the script will:

* Display **Test Accuracy**
* Print **Precision, Recall, F1-Score**
* Show a **Confusion Matrix** (as a heatmap)

---

## 🧠 Model Architecture

* 3 Convolutional Layers
* Max Pooling Layers
* Flatten + Dropout + Dense Layer
* Output Layer with 10 classes (Softmax)

---

## 📚 References

* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* TensorFlow Documentation
* [Kaggle: CIFAR-10 with CNN for Beginners](https://www.kaggle.com/code/roblexnana/cifar10-with-cnn-for-beginer)

---
