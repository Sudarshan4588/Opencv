import cv2

# Load the classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Load an image from the project directory
img = cv2.imread('2.JPG')  # Replace with your image file

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save the result
cv2.imwrite("output.jpg", img)
print(f"Detected {len(faces)} faces. Output saved to output.jpg")
