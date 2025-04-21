# Plant Disease Detection


## Overview

This project implements a deep learning-based plant disease detection system using transfer learning on a VGG16 architecture. The system is designed to identify various plant diseases from leaf images with high accuracy, helping farmers and agricultural specialists detect plant diseases early for timely intervention.

## Dataset

The model is trained on the **Plant Village dataset**, which contains over 87,000 images of healthy and diseased plant leaves across 38 different classes, including:

- Apple (Apple Scab, Black Rot, Cedar Apple Rust, Healthy)
- Blueberry (Healthy)
- Cherry (Powdery Mildew, Healthy)
- Corn/Maize (Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy)
- Grape (Black Rot, Esca/Black Measles, Leaf Blight, Healthy)
- Orange (Huanglongbing/Citrus Greening)
- Peach (Bacterial Spot, Healthy)
- Pepper (Bacterial Spot, Healthy)
- Potato (Early Blight, Late Blight, Healthy)
- Strawberry (Leaf Scorch, Healthy)
- Tomato (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy)

## Model Architecture


The model utilizes a pre-trained VGG16 convolutional neural network with transfer learning:

- **Base Model**: VGG16 pre-trained on ImageNet
- **Transfer Learning**: Feature extraction layers are frozen; only the classifier is trained
- **Fine-tuning**: Custom classifier layers are trained on our specific plant disease dataset
- **Training**: 10 epochs with Adam optimizer and CrossEntropy loss function
- **Performance**: Achieves over 95% validation accuracy

## Results

![Prediction Results](Screenshot%202025-04-21%20233901.png)

The model demonstrates excellent performance in identifying various plant diseases:
- 95.5% validation accuracy across 38 disease classes
- Successfully identifies specific plant diseases even in challenging test scenarios
- Robust performance across different plant species and disease types

## Implementation Details

The complete implementation is contained in the `final_plant.ipynb` Jupyter notebook, which includes:

1. **Data Preparation**:
   - Image loading and preprocessing
   - Data augmentation techniques
   - Dataset splitting (training/validation)

2. **Model Development**:
   - VGG16 architecture adaptation
   - Transfer learning implementation
   - Custom classifier definition

3. **Training Process**:
   - Batch training with Adam optimizer
   - Learning rate management
   - Monitoring loss and accuracy metrics

4. **Evaluation**:
   - Validation on a held-out test set
   - Performance metrics calculation
   - Confusion matrix analysis

5. **Inference**:
   - Real-time disease prediction on new images
   - Visualization of results

## Requirements

- Python 3.6+
- PyTorch (1.7+)
- Torchvision
- NumPy
- Pandas
- Matplotlib
- tqdm
- Jupyter Notebook

## Usage

1. Clone this repository
2. Install the required dependencies
3. Download the Plant Village dataset or use your own plant leaf images
4. Open and run the `final_plant.ipynb` notebook to:
   - Train the model from scratch
   - Evaluate model performance
   - Make predictions on new plant leaf images

## Future Improvements

- Implement ensemble learning for improved accuracy
- Develop a lightweight mobile version for field use
- Add explanation capabilities to provide treatment recommendations
- Extend the model to detect disease severity levels 
