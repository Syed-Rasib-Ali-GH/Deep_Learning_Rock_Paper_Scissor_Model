# 🧩 Deep Learning Test Project — Rock–Paper–Scissors Image Classification using CNN

## 🧠 Project Overview
This project focuses on building and training a **Convolutional Neural Network (CNN)** to classify hand gesture images as **Rock**, **Paper**, or **Scissors**.  
It demonstrates a complete deep learning pipeline — from **data loading** to **model evaluation** — using **TensorFlow/Keras**.

The project was completed as part of a **2-hour deep learning test**, assessing both **practical implementation** and **theoretical understanding** of CNNs.

---

## 📄 Problem Statement
Design and train a CNN model capable of accurately recognizing images representing **rock**, **paper**, or **scissors** hand gestures.  
This problem tests knowledge of:
- CNN architecture and layers  
- Activation and loss functions  
- Model evaluation and overfitting prevention

---

## 🗂️ Dataset Information
**Dataset Name:** Rock–Paper–Scissors Image Dataset  
**Source:** [TensorFlow Datasets - Rock Paper Scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load('rock_paper_scissors', as_supervised=True, with_info=True)
```

The dataset contains labeled images of hand gestures used for the Rock–Paper–Scissors game.

---

## 🎯 Objectives
- Implement a **CNN** to classify images into Rock, Paper, or Scissors.  
- Use **ReLU activation** in hidden layers and **Softmax** in the output layer.  
- Train the model using **Categorical Crossentropy** loss.  
- Evaluate and visualize model performance using metrics and plots.  
- Explain the theoretical reasoning behind key design choices.

---

## 🧩 Tasks to Perform
| Task | Description | Marks |
|------|--------------|-------|
| ✅ **Task 1** | Data Loading and Preprocessing | 15 |
| ✅ **Task 2** | Model Building | 30 |
| ✅ **Task 3** | Model Training | 20 |
| ✅ **Task 4** | Model Evaluation | 15 |
| ✅ **Task 5** | Conceptual Explanation | 20 |

**Conceptual Questions:**
1. Why do we use **ReLU** activation in CNN hidden layers?  
2. Why is **Softmax** used in the output layer?  
3. Why is **Categorical Crossentropy** used as the loss function?  
4. Suggest one method to **reduce overfitting**.

---

## 🧠 Expected Learning Outcomes
- Understand how **CNNs** classify visual data.  
- Justify the use of **activation** and **loss functions** in image classification.  
- Visualize and interpret model **accuracy and loss curves**.  
- Connect **CNN concepts** to real-world computer vision applications.  

---

## 🧰 Technologies Used
- **Python 3**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **TensorFlow Datasets (tfds)**

---

## ⚙️ How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/DL_RPS_Test.git
   cd DL_RPS_Test
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow matplotlib numpy tensorflow-datasets
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook "DL_Test_RockPaperScissors.ipynb"
   ```
4. Follow the notebook steps for data preprocessing, model training, and evaluation.

---

## 📊 Model Evaluation
After training, evaluate model performance using:
- **Accuracy and loss plots**
- **Confusion matrix**
- **Classification report**

---

## 🚀 Future Improvements
- Implement **data augmentation** to reduce overfitting.  
- Try **Transfer Learning** with pretrained CNN models like MobileNet or VGG16.  
- Deploy the trained model as a **web or mobile app**.

---

## 👨‍💻 Author
**Rasib Ali**  
Deep Learning & AI Enthusiast  
[GitHub](https://github.com/rasibali) | [LinkedIn](#)

---

## 🏁 License
This project is open-source and available under the [MIT License](LICENSE).
