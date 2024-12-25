# importing all necessary libraries
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN

'''
 we have used the MTCNN library
'''
# LOAD THE IMAGES FROM THE FILE
image = cv2.imread('face/random.png');

# initializing mtcnn detector
detector = MTCNN()

# detect faces
faces = detector.detect_faces(image)

# drawing boxes around faces
for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(image)
plt.axis('on')
plt.show()