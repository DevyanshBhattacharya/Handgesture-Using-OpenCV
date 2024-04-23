import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv2
import numpy as np
import uuid
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
#mediapipe components
mp_drawing =mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers , callbacks
  





# Live Video using Open Cv
vid = cv2.VideoCapture(0)  
with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as recognizer: # applying mediapipe here 
    while(True): 
      
     
        ret, frame = vid.read() 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting from BGR to RGB as mediapipe reads in rgb
        #image.flags.writable = False #set flag
        
        results = recognizer.process(image) #Makes The DETECTION
     
        #image.flags.writable = True #set flag TO TRUE
        #recognizer = GestureRecognizer(options=options)  # Create an instance of GestureRecognizer
        frame_timestamp_ms = 0  # Define the variable "frame_timestamp_ms"
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #converting back to BGR
        #Rendering Results
        if results.multi_hand_landmarks: #if there is a hand in the frame
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)
                print(num,hand)

        cv2.imshow('frame', image) 
       # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        #recognizer.recognize_async(mp_image, frame_timestamp_ms)  # Use the recognizer instance to call the recognize_async method

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  

frame_timestamp_ms = 0 
vid.release() 

cv2.destroyAllWindows() 

#results.multi_hand_lamnarks            #gives the landmarks of the Previous frame

# Constants
GESTURE_CATEGORIES = {'Rock': 0, 'Kartik':2, 'Thumbs UP':3,'Left':4}
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720  # Adjust based on your requirements
DATA_DIR = "C:\Vs Code\Machine learning\OpenCv\Handgesture\GESTURE_CATEGORIES"
CSV_FILENAME = "handLandmarks.csv"
MODEL_FILENAME = "handtracking.h5"

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory '{DATA_DIR}' not found.")
    exit()

# Function to extract hand landmarks using Mediapipe
def extract_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(rgb_image)

    landmarks = []
    if results.multi_hand_landmarks: # if it can detect a hand 
        for hand_landmarks in results.multi_hand_landmarks: #starts reading it and assign it x,y,z coordinates
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

    hands.close()
    return landmarks

# Load images, extract hand landmarks, and save data to CSV
data = {"landmarks": [], "label": []}  # Defined a Dictionary

for label in (GESTURE_CATEGORIES):
    category_dir = os.path.join(DATA_DIR, str(label))

    if not os.path.exists(category_dir):
        print(f"Error: Category directory '{category_dir}' not found.")
        exit()

    for image_name in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        landmarks = extract_hand_landmarks(image)
        data["landmarks"].append(landmarks)
        data["label"].append(GESTURE_CATEGORIES[label])

# Convert the data dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.dropna(inplace=True)
df.to_csv(CSV_FILENAME, index=False)
print(f"Hand landmarks data saved to '{CSV_FILENAME}'.")



# Constants
GESTURE_CATEGORIES = 4

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory '{DATA_DIR}' not found.")
    exit()

# Function to extract hand landmarks using Mediapipe
def extract_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(rgb_image)

    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

    hands.close()
    return landmarks

# Load images, extract hand landmarks, and save data to CSV
data = {"landmarks": [], "label": []}
print(f"Reading ... '{CSV_FILENAME}'.")
df = pd.read_csv(CSV_FILENAME)



# Extract features (landmarks) and labels

X = np.array([np.fromstring(x[1:-1], sep=',', dtype=float).reshape(-1, 3) for x in df['landmarks']])


y = df['label'].values


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)


# print(len(X_train[0]))
# Define the Keras model
model = keras.models.Sequential([
    keras.layers.Input(shape=(21, 3), dtype='float32'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dropout(0.5),
    keras.layers.Dense(8, activation='softmax')
])

X_train = X_train.astype(float)
X_test = X_test.astype(float)
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[early_stopping])
# Train the model
# model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Save the trained model
model.save(MODEL_FILENAME)
print(f"Model saved to '{MODEL_FILENAME}'.")


GESTURE_CATEGORIES = {'Rock': 0, 'Kartik':2, 'Thumbs UP':3,'Left':4}
# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 1280 ,720


# Load the saved Keras model
MODEL_FILENAME = "handtracking.h5"
loaded_model = keras.models.load_model(MODEL_FILENAME)

# Function to extract hand landmarks using Mediapipe
def extract_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(rgb_image)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

    hands.close()
    return landmarks


# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Extract hand landmarks
    # Extract hand landmarks
    landmarks = extract_hand_landmarks(frame)
    if not landmarks:
        print("No hand landmarks detected.")
        continue
    # Process landmarks for prediction
    landmarks = np.array([landmarks])
    landmarks = landmarks.reshape(landmarks.shape[0], 21, 3)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(landmarks)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(landmarks)

    # Get the predicted class
    predicted_class = np.argmax(predictions)
    for key in GESTURE_CATEGORIES :
        if (GESTURE_CATEGORIES[key]==predicted_class):

    # Display the predicted class on the frame
            cv2.putText(frame, f'Gesture: {key}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()