# Speech Emotion Classification Model

In this project, I utilized the RAVDESS public dataset, which includes eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, and surprised) expressed at two intensities: normal and strong. I successfully built a classification model, extracting five features from speech signals to discern between different emotions. The model provided valuable insights into classifying emotional states in speech.

## EDA
The first step is to understand the type of data we are working with.

### Distribution of data for each emotion

The plot e below displays the distribution of emotions in a dataset. Each emotion occurs 192 times, except for "neutral," which appears 96 times. The dataset is relatively balanced across emotions.
<img src="./EDA/Count for each emotions.png"/>

### Waveform, spectrogram and F0 for data

Below, you will find three different plots for two speech emotions: 'Calm' and 'Happy

#### For speech signal with “Calm” label:

<img src="./EDA/waveform_calm.png"/>
<img src="./EDA/spec_calm.png"/>
<img src="./EDA/f0_calm.png"/>

#### For speech signal with “Happy” label:
<img src="./EDA/wave_happy.png"/>
<img src="./EDA/spec_happy.png"/>
<img src="./EDA/F0_happy.png"/>

## Data Augmentation

I have implemented several data augmentation functions to enhance the diversity of my dataset. The **`noise`** function introduces random amplitude noise to the input data, contributing variability in signal strength. For temporal transformations, the **`shift`** function horizontally displaces the data by a random amount within a specified range, simulating temporal shifts. The **`time_scale`** function performs time scaling on the data, altering its temporal structure by resampling. Additionally, the **`bandpass_filter`** function applies a bandpass filter to focus on specific frequency components within the data, providing resilience against noise outside the desired range. Collectively, these augmentation techniques aim to improve my model's robustness by exposing it to a more varied set of patterns and features present in the data.

## Before Augmentation

The count of data points in the dataset before augmentation:

- calm: 192
- happy: 192
- sad: 192
- angry: 192
- fear: 192
- disgust: 192
- surprise: 192
- neutral: 96

## After Augmentation

The count of data points in the dataset after augmentation:

- calm: 960
- happy: 960
- sad: 960
- angry: 960
- fear: 960
- disgust: 960
- surprise: 960
- neutral: 480

## Feature Extraction

Five features were extracted from the speech signal:

1. Zero Crossing Rate (ZCR)
2. Mel-frequency cepstral coefficients (MFCC)
3. Chroma Short-Time Fourier Transform (Chroma STFT)
4. Root Mean Square Value (RMS)
5. Fundamental Frequency (F0)

1. **Zero Crossing Rate (ZCR):** A scalar value, so its shape is (1,).
2. **MFCC (Mel-frequency cepstral coefficients):** Typically, a vector with a variable number of coefficients, so its shape is be (n_mfcc,).
3. **Chroma STFT:** A vector with a variable number of coefficients, so its shape is (n_chroma,).
4. **Root Mean Square Value (RMS):** A scalar value, so its shape is (1,).
5. **F0 (Fundamental Frequency):** A scalar value, so its shape is (1,).

When these features are concatenated the shape of the resulting array will be the sum of the individual shapes along the horizontal axis. Therefore, the shape of output is (35,1)


## Model

2 models were build for this project to demonestrate the effect of data augmentation on the output.

### 1. ANN model without data augmentation

1. **Oversampling of 'neutral' Target Value:**
    - To tackle the class imbalance issue, I performed oversampling of the 'neutral' target value. This ensured that the model was exposed to a balanced representation of each class during training, ultimately improving its ability to generalize.
2. **One-Hot Encoding for Target Value:**
    - I applied one-hot encoding to the target variable, converting categorical labels into a binary matrix format. This transformation was crucial for compatibility with the model's output layer, which utilizes the softmax activation function.
3. **Train-Test Split:**
    - I divided the dataset into training and testing sets, a standard practice to evaluate the model's performance on unseen data and prevent overfitting.
4. **Feature Standardization with StandardScaler():**
    - I employed the StandardScaler() to standardize the input features. This standardization ensured that all features had a mean of 0 and a standard deviation of 1, preventing certain features from dominating the model training process.
5. **ANN Model Architecture:**
    - I constructed an ANN with a specific architecture, including:
        - Input Layer: Matching the number of neurons to the input feature dimension.
        - Hidden Layer 1: 256 neurons with the ReLU activation function.
        - Hidden Layer 2: 128 neurons with the ReLU activation function.
        - Hidden Layer 3: 32 neurons with the ReLU activation function.
        - Output Layer: The number of neurons equaled the number of classes, utilizing the softmax activation function for multiclass classification.
    - I set the learning rate to 0.0001, the number of epochs to 100, and a batch size of 30 for efficient training. I employed the Adam optimizer and categorical crossentropy as the loss function, suitable for multiclass classification tasks.
  

## Result for ANN model without data augmentation
1. Training and Validation Accuracy and Training and Validation Loss Before Data Augmentation:

<img src="./outputs/val_test_loss_no_aug.png"/>

2. Confusion Matrix:

<img src="./outputs/confmat_no_aug.png"/>












