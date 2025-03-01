from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from deepface import DeepFace
import os

app = Flask(__name__)

# Function to create the model if loading fails
def create_ocean_model():
    model = Sequential([
        Dense(16, input_dim=6, activation='relu', name='input_layer'),  # Input: one-hot encoded emotions (6 categories)
        Dense(8, activation='relu'),
        Dense(5, activation='sigmoid')  # Output: OCEAN traits (5 traits)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Try to load the model, recreate if fails
try:
    model = tf.keras.models.load_model('finalocean.h5')
    print("Existing model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Creating a new model...")
    model = create_ocean_model()
    
    # Sample training data
    emotions = ['happy', 'surprise', 'neutral', 'sad', 'anger', 'fear']
    inputs = np.array([
        [1, 0, 0, 0, 0, 0],  # happy
        [0, 1, 0, 0, 0, 0],  # surprise
        [0, 0, 1, 0, 0, 0],  # neutral
        [0, 0, 0, 1, 0, 0],  # sad
        [0, 0, 0, 0, 1, 0],  # anger
        [0, 0, 0, 0, 0, 1],  # fear
    ])
    outputs = np.array([
        [1, 0, 1, 1, 0],  # OCEAN for 'happy'
        [1, 0, 1, 1, 0],  # OCEAN for 'surprise'
        [0, 1, 0, 1, 0],  # OCEAN for 'neutral'
        [0, 1, 0, 1, 1],  # OCEAN for 'sad'
        [0, 0, 0, 0, 1],  # OCEAN for 'anger'
        [0, 0, 0, 0, 1],  # OCEAN for 'fear'
    ])
    
    # Train the model with validation split
    model.fit(inputs, outputs, epochs=100, verbose=1, validation_split=0.2)
    
    # Save the newly trained model
    model.save('finalocean.h5')
    print("New model trained and saved.")

# Emotion to OCEAN mapping (for fallback)
emotion_to_ocean = {
    'happy': {'Openness': 0.7, 'Conscientiousness': 0.6, 'Extraversion': 0.8, 'Agreeableness': 0.7, 'Neuroticism': 0.3},
    'sad': {'Openness': 0.4, 'Conscientiousness': 0.5, 'Extraversion': 0.3, 'Agreeableness': 0.6, 'Neuroticism': 0.8},
    'anger': {'Openness': 0.5, 'Conscientiousness': 0.6, 'Extraversion': 0.4, 'Agreeableness': 0.2, 'Neuroticism': 0.9},
    'surprise': {'Openness': 0.8, 'Conscientiousness': 0.5, 'Extraversion': 0.7, 'Agreeableness': 0.6, 'Neuroticism': 0.4},
    'neutral': {'Openness': 0.5, 'Conscientiousness': 0.5, 'Extraversion': 0.5, 'Agreeableness': 0.5, 'Neuroticism': 0.5},
    'fear': {'Openness': 0.4, 'Conscientiousness': 0.6, 'Extraversion': 0.3, 'Agreeableness': 0.2, 'Neuroticism': 0.9},
}

# Store aggregated OCEAN scores
aggregated_scores = []

def map_emotion_to_ocean(dominant_emotion):
    try:
        # Convert one-hot encoded emotion to OCEAN traits using the model
        emotion_vector = np.array([map_emotion_to_vector(dominant_emotion)])
        ocean_traits = model.predict(emotion_vector)[0]
        
        return {
            'Openness': ocean_traits[0],
            'Conscientiousness': ocean_traits[1],
            'Extraversion': ocean_traits[2],
            'Agreeableness': ocean_traits[3],
            'Neuroticism': ocean_traits[4]
        }
    except Exception as e:
        print(f"Model prediction failed: {e}")
        # Fallback to predefined mapping
        return emotion_to_ocean.get(dominant_emotion, {
            'Openness': 0.5,
            'Conscientiousness': 0.5,
            'Extraversion': 0.5,
            'Agreeableness': 0.5,
            'Neuroticism': 0.5
        })

def map_emotion_to_vector(emotion):
    """Convert emotion to a one-hot vector."""
    emotions = ['happy', 'surprise', 'neutral', 'sad', 'anger', 'fear']
    return [1 if emotion == e else 0 for e in emotions]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global aggregated_scores
        cap = cv2.VideoCapture(0)
        # Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]

                try:
                    # Use DeepFace to detect emotion
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    dominant_emotion = analysis.get('dominant_emotion', None)
                    
                    if dominant_emotion:
                        # Map emotion to OCEAN traits
                        ocean_scores = map_emotion_to_ocean(dominant_emotion.lower())
                        aggregated_scores.append(ocean_scores)

                        # Display personality ratios on frame
                        y_offset = 0
                        for trait, score in ocean_scores.items():
                            text = f"{trait}: {score:.2f}"
                            cv2.putText(frame, text, (x, y-10-y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                            y_offset += 20

                except Exception as e:
                    print(f"Error analyzing face: {e}")

            # Encode the frame to send to the client
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze():
    global aggregated_scores
    if aggregated_scores:
        # Calculate average OCEAN scores
        final_scores = {trait: np.mean([score[trait] for score in aggregated_scores]) for trait in aggregated_scores[0].keys()}
        
        # Create pie chart for results
        create_pie_chart(final_scores)
        
        aggregated_scores = []  # Clear scores for next session
        return redirect(url_for('results'))
    else:
        return "No data captured. Please try again.", 400

def create_pie_chart(ocean_scores):
    labels = list(ocean_scores.keys())
    sizes = list(ocean_scores.values())
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Average OCEAN Personality Traits', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('static/ocean_pie_chart.png')
    plt.close()

@app.route('/results')
def results():
    return render_template('results.html', image_url='/static/ocean_pie_chart.png')

if __name__ == '__main__':
    # Ensure static and template directories exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)