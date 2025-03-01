# AI-Based Psychometric Test

## Overview  
Psychometric evaluations play a crucial role in personality assessment, particularly in recruitment and personal development. Traditional methods suffer from manual errors, delays, and lack of real-time analysis. This project integrates **Artificial Intelligence (AI)** to enhance the accuracy and efficiency of psychometric testing using a **dual-modality approach**:  
- **Text-based MBTI classification** using **BERT** and **Logistic Regression**  
- **Video-based OCEAN personality trait prediction** using **OpenCV** and **DeepFace**  

This AI-driven framework achieves **88.7% accuracy** for MBTI predictions and **93.5% accuracy** for OCEAN trait predictions.  

##  Objectives  
- Develop an **AI-powered psychometric assessment** system.  
- Implement **BERT** for text-based **MBTI classification**.  
- Utilize **OpenCV** and **DeepFace** for real-time **video-based personality analysis**.  
- Achieve **high accuracy** in personality classification using AI.  
- Provide **real-time insights** into personality traits.  

## Methodology  

### **Text-Based Personality Classification (MBTI)**  
- Uses **MBTI dataset from Kaggle** (~8600 records).  
- **Preprocessing:** Tokenization, stopword removal, and normalization.  
- **Feature Extraction:** BERT transformer model for **contextual embeddings**.  
- **Classification Model:** Logistic Regression trained on **80-20 train-test split**.  
- **Accuracy:** **88.7%**  

### **Video-Based Personality Analysis (OCEAN Traits)**  
- Uses **real-time video** for personality assessment.  
- **Facial Analysis:** OpenCV captures video, and DeepFace analyzes expressions.  
- **Emotion Mapping:** Maps detected emotions to **Big Five (OCEAN) traits**.  
- **CNN Model:** Trained on emotion recognition datasets.  
- **Accuracy:** **93.5%**  


##  Implementation Details  
- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - NLP: `Transformers`, `BERT`, `Scikit-learn`  
  - Video Processing: `OpenCV`, `DeepFace`  
  - Deep Learning: `TensorFlow`, `PyTorch`  
- **Deployment:** Web-based UI for user input and analysis visualization.  


##  Results  
- **Text-Based MBTI Classification Accuracy:** **88.7%**  
- **Video-Based OCEAN Personality Traits Accuracy:** **93.5%**  
- **Real-time analysis** improves **consistency** and **reduces subjective bias**.  
