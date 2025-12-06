# 📌 CIFAR-10 Image Classification using CNN | Custom Augmentations + Grad-CAM 🔥

Welcome to my deep learning project focused on **image classification using the CIFAR-10 dataset**!
This project explores **Convolutional Neural Networks (CNNs)**, **data augmentation**, and **model interpretability** techniques to achieve higher accuracy and meaningful visual explanations.

---

## 🚀 Project Highlights

✔ Built & trained a **custom CNN** from scratch on CIFAR-10
✔ Achieved **~89% accuracy** on test data
✔ Implemented **advanced data augmentations**:

* **CutMix** 🧩
* **MixUp** 🎨

✔ Added **Grad-CAM** 🔎 to visualize what the model “sees”
✔ Designed a **modular architecture** with reusable scripts
✔ Fully reproducible training pipeline using PyTorch

---

## 🧠 What is CIFAR-10?

CIFAR-10 is a widely used computer vision dataset containing:

| 📷 Images | 🏷 Classes | Dimensions  |
| --------- | ---------- | ----------- |
| 60,000    | 10         | 32 × 32 RGB |

Example label classes include: **airplane, car, bird, cat, deer, dog, frog, horse, ship, truck**.

---

## 🏗️ Project Structure

```
Image_classifyCIFAR-10/
│
├── src/
│   ├── dataset.py       # Loads CIFAR-10 with augmentations
│   ├── model.py         # Custom CNN architecture
│   ├── utils.py         # Accuracy, loss + augmentation helpers
│   ├── train.py         # Training + validation loop
│   ├── gradcam.py       # Grad-CAM visualization
│   └── config.py        # Hyperparameters & settings
│
├── README.md            # ✨ You are here!
└── requirements.txt     # Required libraries
```

---

## 🏁 Training Results

📈 Final Model Performance:

* **Accuracy:** ~89%
* **Loss curve and logs shown during training**
* Improvement achieved through **custom data augmentations**

---

## 🔥 Grad-CAM Visualizations

**Why Grad-CAM?**
It helps reveal *where the model is looking* while making predictions — enhancing trust and explainability.

> The Grad-CAM heatmaps highlight the most relevant regions of an image that influenced the prediction.

Example (Input → Heatmap → Overlay):
➡🚀 (Add your generated Grad-CAM result images here!)

---

## 📚 Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **OpenCV**
* **Matplotlib**
* **NumPy**

---

## 📚 Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **OpenCV**
* **Matplotlib**
* **NumPy**

---

## 🌟 What I Learned

✔ Practical experience with CNNs
✔ Importance of augmentations in boosting model performance
✔ Using Grad-CAM for **model interpretability**
✔ Working with **virtual environments + GitHub version control**

---

## 🚧 Future Improvements

* Experiment with **ResNet / EfficientNet** for >92% accuracy
* Deploy model as a **web app** using Streamlit or Flask
* Include real-time webcam inference

---

## 🤝 Contributions & Feedback

This project is part of my **Machine Learning learning journey**.
Suggestions and feedback are **always welcome**!
Feel free to open issues or contribute through PRs. 🚀

---

## 👩‍💻 Author

**Namit Kumar**
📌 Deep Learning Enthusiast
📌 Exploring model interpretability and real-world applications
