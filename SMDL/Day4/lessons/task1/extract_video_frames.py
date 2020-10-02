# Video frame extraction using OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SEQUENCE_LEN = 20
STEP = 3  # frames
IMAGE_SIZE = (160, 160)  # width, height

images = []
count = 0
cap = cv2.VideoCapture('video.mov')

while cap.isOpened() and len(images) < SEQUENCE_LEN:
    ret, frame = cap.read()
    if not ret:
        print(f'No more frames from cap.read')
        break

    # only take every STEP frame
    if count % STEP == 0:
        # resize with linear interpolation
        resized = cv2.resize(frame, IMAGE_SIZE,
                             interpolation=cv2.INTER_LINEAR)
        images.append([preprocess_input(resized)])
    count += 1

# cleanup opencv stuff
cap.release()
cv2.destroyAllWindows()

# stack the images along the batch dimension (axis=0)
images = np.stack(images)
print(images.shape)

# plot first 10 frames
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 10))
axes = axes.flatten()
for i in range(min(len(images), len(axes))):  # only loop up to the number of plots or images

    # 1. preprocess_input will add a batch dimension (1, h, w, channels)
    #    but matplotlib only understands (h, w, channels)
    #
    # 2. preprocess_input will also do zero-mean centering,
    #    but matplotlib only understands [0..] for floats
    #
    axes[i].imshow((images[i][0] * .5) + .5)
    axes[i].set_title(f'frame {i*STEP}')

fig.suptitle('Guess the activity?')
plt.savefig('frames.png')
plt.show()
