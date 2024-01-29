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
