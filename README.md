# Celebrity Face Matching App

A real-time face matching application built with Streamlit that compares webcam input to celebrity faces using computer vision and machine learning techniques.

## Overview

This application uses your webcam to detect faces in real-time and matches them against a database of celebrity faces, showing you which celebrities you most resemble with confidence percentages.

## Features

- **Real-time face detection** using webcam input
- **Celebrity matching** with confidence percentages
- **Interactive configuration** of model parameters
- **Responsive UI** with live video feed and match results

## Technical Components

### 1. User Interface

- **Two-column layout**:
  - Left: Live video feed with face detection
  - Right: Top 5 matches with confidence percentages
- **Interactive controls** for model parameters
- **Visual feedback** with bounding boxes and progress bars

### 2. Model Configuration

- **SVM Parameters**:
  - Kernel type selection (linear, rbf, poly)
  - Regularization parameter (C)
  - Probability estimation toggle
- **Feature Extraction Settings**:
  - HOG descriptor window size
  - Block size and stride
  - Cell size and number of bins
- **Processing Options**:
  - Frame skip rate
  - Face change threshold

### 3. Face Detection & Processing

- WebRTC for efficient video streaming
- Haar cascade classifiers for face detection
- Frame skipping for performance optimization
- Change detection to avoid redundant processing

### 4. Feature Extraction

- HOG (Histogram of Oriented Gradients) implementation
- Dynamic feature vector calculation
- Image preprocessing pipeline (grayscale conversion and resizing)

### 5. Model Management

- Pre-trained models downloaded from Google Drive
- Version mismatch handling with warning suppression
- Resource caching for improved performance
- Fallback mechanisms for model loading failures

## Technical Stack

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision operations
- **scikit-learn**: Machine learning (SVM classifier)
- **WebRTC**: Real-time video streaming
- **NumPy**: Numerical operations
- **gdown**: Google Drive integration

## Development Process

The development process included:

### Data Preparation
- Loading and processing celebrity face datasets
- Face detection and cropping
- Data augmentation techniques
- Train-test splitting

### Feature Engineering
- HOG parameter experimentation
- Feature vector analysis
- Dimensionality reduction techniques

### Model Training
- SVM classifier training with different kernels
- Hyperparameter tuning
- Cross-validation and evaluation

### Deployment
- Model export and optimization
- Streamlit application development
- Performance optimization

## Future Improvements

- Add face registration functionality
- Implement model retraining capability
- Add support for multiple face detection
- Improve error handling for edge cases
- Add performance metrics display
- Enhance UI with additional visualizations

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install system dependencies: `apt-get install $(cat packages.txt)`
4. Run the application: `bash app.sh` or `streamlit run Face_App.py`

## Requirements

See `requirements.txt` for Python dependencies and `packages.txt` for system dependencies.

# JUPYTER NOTEBOOK .ipynb file:
1. Data Preparation

    Loading face datasets (possibly LFW, CelebA, or custom dataset)

    Face detection and cropping

    Data augmentation techniques

    Train-test splitting

2. Feature Extraction

    HOG parameter experimentation

    Feature vector analysis

    Dimensionality reduction (PCA, LDA)

3. Model Training

    SVM classifier training with different kernels

    Hyperparameter tuning (GridSearchCV)

    Cross-validation results

    Model evaluation metrics

4. Model Export

    Saving trained model to pickle files

    Exporting label encoder and dictionary

    Model size optimization
