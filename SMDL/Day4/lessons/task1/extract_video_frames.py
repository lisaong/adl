# Video frame extraction using OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# http://futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('seaborn')

SEQUENCE_LEN = 20
START_OFFSET = 10  # frames
STEP = 3  # frames
IMAGE_SIZE = (128, 128)  # width, height

images = []
count = 0
cap = cv2.VideoCapture('video.mov')

while cap.isOpened() and len(images) < SEQUENCE_LEN:
    ret, frame = cap.read()
    if not ret:
        print(f'No more frames from cap.read')
        break

    # after START_OFFSET, take every STEP frame
    if count > START_OFFSET and count % STEP == 0:
        # convert BGR format to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize with linear interpolation
        frame = cv2.resize(frame, IMAGE_SIZE,
                           interpolation=cv2.INTER_LINEAR)
        images.append([preprocess_input(frame)])
    count += 1

# cleanup opencv stuff
cap.release()
cv2.destroyAllWindows()

# stack the images along the batch dimension (axis=0)
images = np.vstack(images)
print(f'Extracted frames shape: {images.shape}')

# plot first 10 frames
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 8))
axes = axes.flatten()
for i in range(min(len(images), len(axes))):  # only loop up to the number of plots or images
    # mobilenet_v2.preprocess_input will do zero-mean centering,
    # but matplotlib only understands [0..1] for floats
    axes[i].imshow((images[i] * .5) + .5)
    axes[i].set_title(f'frame {START_OFFSET + i*STEP}')

fig.suptitle('Guess the activity')
plt.savefig('frames.png')
plt.show()
