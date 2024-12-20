import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the model
model = MobileNetV2(weights="imagenet")
print("Modello caricato con successo :)")


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore: Impossibile accedere alla webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from webcam
    if not ret:
        print("Errore: Frame non disponibile.")
        break

    # Prepare the frame for the model
    resized_frame = cv2.resize(frame, (224, 224))  # Adapt to the dimension of the model
    input_data = preprocess_input(np.expand_dims(resized_frame, axis=0))  # Normalize the data

    # Make a prevision
    predictions = model.predict(input_data)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Main result
    label = f"{decoded_predictions[0][1]}: {decoded_predictions[0][2]:.2f}"

    # Show the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the feed of the webcam
    cv2.imshow("Webcam Feed", frame)

    # Tap 'q' in order to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
