from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle
from mtcnn.mtcnn import MTCNN
"""
import cv2
import numpy as np
from imaage_preprocessing import load_images_from_directory
"""


# draw the shape with detected object
def draw_iamge_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing the boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill = False, color = 'red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius = 2, color = 'red')
            ax.add_patch(dot)

    # show the plot
    pyplot.show()


"""
# load the directory
directory_dataset = "face/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart"
data, labels = load_images_from_directory(directory_dataset)
"""
file_name = "face/OIP.jpeg"
pixels = pyplot.imread(file_name)

# create a detector
detector = MTCNN()

# detect faces in the image
faces = detector.detect_faces(pixels)
for face in faces:
    print(face)

# dispaly faces of the original photo
draw_iamge_with_boxes(file_name, faces)

"""
for i, image in enumerate(data):
    # convert the image back to 3 channels to use in MDNN
    image_3ch = cv2.cvtColor((image * 255.0), cv2.COLOR_GRAY2RGB)
    faces = detector.detect_faces(image_3ch)

    # display the image with boxes
    print(f"Detected faces for {labels[i]}:")
    draw_iamge_with_boxes(image_3ch, faces)
"""
