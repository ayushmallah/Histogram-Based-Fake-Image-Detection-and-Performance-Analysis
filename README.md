# Histogram-Based-Fake-Image-Detection-and-Performance-Analysis
This project focuses on detecting whether an image is real or fake using histogram analysis and machine learning techniques. The system analyzes pixel intensity distribution patterns in images and identifies inconsistencies commonly found in manipulated or AI-generated images.

# Project Overview
With the rapid growth of image editing tools and AI-generated content, distinguishing between real and fake images has become challenging. This project provides a lightweight and efficient solution using image histograms as features instead of heavy deep learning models.

The application allows users to upload an image, processes it, and predicts whether it is authentic or manipulated.

# How It Works
The input image is loaded and preprocessed.
A histogram is generated based on pixel intensity values (grayscale or RGB).
Histogram features are extracted and converted into a feature vector.
A trained machine learning model (e.g., Naive Bayes / SVM / Logistic Regression) is used for classification.
The system outputs whether the image is Real or Fake, along with probability scores.

# Technologies Used
* Python
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Tkinter (for GUI)
* Joblib (for model saving/loading)

# Features
* Histogram-based feature extraction
* Lightweight and fast detection
* Simple graphical user interface
* Model evaluation using confusion matrix and ROC curve
* Report generation functionality

# Dataset
The dataset consists of two categories:

Real Images
Fake/Manipulated Images

Images are used to train and test the model for classification.

# Results
The model is trained to distinguish patterns in histogram distributions. Fake images often show unnatural pixel distributions compared to real images, which helps in accurate classification.

# Limitations
* Accuracy depends on dataset quality
* May not detect highly advanced deepfake images
* Works best with controlled datasets

# Future Scope
* Integration with deep learning (CNN models)
* Real-time detection using camera input
* Web-based deployment
* Improved accuracy with larger datasets

# Author
Developed as part of a major project in the field of image forensics and machine learning.
