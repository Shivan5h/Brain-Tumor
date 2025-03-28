# Brain Tumor Classification Using EfficientNetB0

## Overview  
This project classifies brain tumors into four categories using deep learning. The **EfficientNetB0** architecture is used as the feature extractor, and the model is fine-tuned to improve accuracy. The dataset is preprocessed using **ImageDataGenerator**, and training is performed using **TensorFlow/Keras** with GPU acceleration (if available).   

## Dataset  
The dataset consists of MRI scan images categorized into four classes:
- **glioma**  
- **meningioma**  
- **notumor**  
- **pituitary**  

### Directory Structure  
```
img/
│── Training/     # Contains training images
│── Testing/      # Contains testing images
```

## Dependencies  
Ensure you have the required libraries installed:  
```bash  
pip install tensorflow numpy opencv-python scikit-learn  
```

## Code Explanation  

### 1. Importing Required Libraries  
```python  
import os  
import numpy as np  
import cv2  
import tensorflow as tf  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.applications import EfficientNetB0  
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.optimizers import AdamW  
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  
from sklearn.utils.class_weight import compute_class_weight  
from sklearn.metrics import classification_report, accuracy_score  
```
- **TensorFlow & Keras**: Used for model creation and training.  
- **EfficientNetB0**: A pre-trained deep learning model that is fine-tuned for classification.  
- **ImageDataGenerator**: Used for data augmentation.  
- **AdamW Optimizer**: An adaptive optimizer used for better training convergence.  
- **EarlyStopping & ReduceLROnPlateau**: Callbacks to stop training if performance stagnates.  
- **compute_class_weight**: Handles class imbalance by assigning appropriate weights.  

### 2. Enabling GPU for Faster Training  
```python  
physical_devices = tf.config.experimental.list_physical_devices('GPU')  
if physical_devices:  
    for device in physical_devices:  
        tf.config.experimental.set_memory_growth(device, True)  
    print("GPU is available and will be used.")  
else:  
    print("GPU is not available, running on CPU.")  
```
- Ensures TensorFlow uses GPU if available.  
- If no GPU is detected, training falls back to CPU.  

### 3. Data Preprocessing & Augmentation  
```python  
datagen_train = ImageDataGenerator(  
    rescale=1./255,  
    rotation_range=30,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.3,  
    horizontal_flip=True,  
    brightness_range=[0.7, 1.3],  
    fill_mode='nearest'  
)
```
- **rescale=1./255**: Normalizes pixel values between 0 and 1.  
- **rotation_range, zoom_range**: Introduces slight variations in images to make the model robust.  
- **horizontal_flip**: Helps model generalization.  
- **brightness_range**: Adjusts brightness randomly.  

```python  
datagen_test = ImageDataGenerator(rescale=1./255)  
```
- Only normalizes the test data (no augmentation).  

### 4. Loading the Dataset  
```python  
train_data = datagen_train.flow_from_directory(  
    train_dir,  
    target_size=img_size,  
    batch_size=batch_size,  
    class_mode='sparse',  
    shuffle=True  
)
```
- Reads images from `train_dir`, resizes them, and assigns labels.  
- **class_mode='sparse'**: Uses sparse integer labels instead of one-hot encoding.  

```python  
test_data = datagen_test.flow_from_directory(  
    test_dir,  
    target_size=img_size,  
    batch_size=batch_size,  
    class_mode='sparse',  
    shuffle=False  
)
```
- Loads test images without shuffling for proper evaluation.  

### 5. Handling Class Imbalance  
```python  
class_weights = compute_class_weight(  
    class_weight='balanced',  
    classes=np.unique(train_data.classes),  
    y=train_data.classes  
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}  
```
- Assigns higher weights to underrepresented classes to balance learning.  

### 6. Building the Model  
```python  
model = Sequential([  
    EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),  
    GlobalAveragePooling2D(),  
    BatchNormalization(),  
    Dense(128, activation='relu'),  
    Dropout(0.3),  
    Dense(len(categories), activation='softmax')  
])  
```
- Uses EfficientNetB0 as a feature extractor.  
- Applies **BatchNormalization** for stable training.  
- Adds **Dropout (0.3)** to reduce overfitting.  
- The final layer has **softmax activation** for multi-class classification.  

### 7. Compiling the Model  
```python  
model.compile(  
    optimizer=AdamW(learning_rate=1e-4),  
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']  
)  
```
- **AdamW**: Optimizer for efficient weight updates.  
- **Sparse Categorical Crossentropy**: Used because labels are integers.  

### 8. Training the Model  
```python  
model.fit(  
    train_data,  
    validation_data=test_data,  
    epochs=20,  
    class_weight=class_weights_dict,  
    callbacks=[  
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)  
    ]  
)  
```
- **EarlyStopping**: Stops training if no improvement in `val_loss`.  
- **ReduceLROnPlateau**: Lowers the learning rate when training stagnates.  

### 9. Model Evaluation  
```python  
y_true = test_data.classes  
y_pred = np.argmax(model.predict(test_data), axis=1)  
print(classification_report(y_true, y_pred))  
print("Accuracy:", accuracy_score(y_true, y_pred))  
```
- **Predicts on the test set** and evaluates performance.  

## Running the Code  
Execute the script:  
```bash  
python code.py  
```

## Output Example  
```
Classification Report:
              precision    recall  f1-score   support
      glioma       0.85      0.83      0.84       XXX
 meningioma       0.89      0.87      0.88       XXX
    notumor       0.95      0.98      0.96       XXX
  pituitary       0.90      0.89      0.89       XXX
Accuracy: 0.90
```

## Notes  
- Ensure the dataset path is correct.  
- Modify hyperparameters as needed for better performance.  

