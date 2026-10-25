# EEG-BASED-HUMAN-EMOTION-RECOGNITION-USING-DEEP-LEARNING
## 📌 Overview
This project aims to classify human emotions from EEG (Electroencephalogram) signals using Deep Learning techniques. The model analyzes brainwave patterns to recognize emotional states like Happy, Sad, Angry, etc.

This was developed as part of B.Tech CSE final year project.

## 🧠 Dataset
- **DEAP Dataset**: A Database for Emotion Analysis using Physiological Signals
- Link: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- Contains EEG signals and emotional ratings (Valence, Arousal, Dominance)

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## 🔧 Methodology
1.  **Data Preprocessing**: Filtering, noise removal, and normalization of EEG signals
2.  **Feature Extraction**: Extracting relevant features from EEG bands (Alpha, Beta, Theta, Delta)
3.  **Model Building**: Built CNN-based Deep Learning model for emotion classification
4.  **Training & Evaluation**: Trained on DEAP dataset and evaluated performance

## 📊 Results
- Explored different preprocessing techniques and model architectures
- Model trained for emotion classification based on Valence-Arousal model
- Focused on improving accuracy for Human-Computer Interaction and Healthcare applications

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/Nandinireddy23/EEG-BASED-HUMAN-EMOTION-RECOGNITION-USING-DEEP-LEARNING.git

# Install dependencies
pip install numpy pandas matplotlib tensorflow scikit-learn

# Run the model
python main.py
