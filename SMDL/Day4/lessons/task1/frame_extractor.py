# Video frame extraction using OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# http://futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('seaborn')


def extract_frames(video_path, sequence_len, start_offset=0, step=1,
                   image_size=(128, 128)):
    imgs = []
    ids = []

    count = 0
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and len(imgs) < sequence_len:
        ret, frame = cap.read()
        if not ret:
            print(f'No more frames from cap.read')
            break

        # after start_offset, take every step frame
        if count > start_offset and count % step == 0:
            # convert BGR format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize with linear interpolation
            frame = cv2.resize(frame, image_size,
                               interpolation=cv2.INTER_LINEAR)
            imgs.append([preprocess_input(frame)])
            ids.append(count)
        count += 1

    # cleanup opencv stuff
    cap.release()
    cv2.destroyAllWindows()

    # stack the images along the batch dimension (axis=0)
    return np.vstack(imgs), ids


def plot_images(imgs, ids=None, title='', rows=2, cols=5,
                output_filename='frames.png'):
    # plot first nrows*ncols frames
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 8))
    axes = axes.flatten()
    for i in range(min(len(imgs), len(axes))):  # only loop up to the number of plots or images
        # mobilenet_v2.preprocess_input will do zero-mean centering,
        # but matplotlib only understands [0..1] for floats
        axes[i].imshow((imgs[i] * .5) + .5)
        if ids:
            axes[i].set_title(f'frame {ids[i]}')
        else:
            axes[i].set_title(f'{i}')

    fig.suptitle(title)
    plt.savefig(output_filename)
    plt.show()


if __name__ == "__main__":

    SEQUENCE_LEN = 20
    START_OFFSET = 10  # frames
    STEP = 3  # frames
    IMAGE_SIZE = (128, 128)  # width, height

    images, frame_ids = extract_frames('video.mov', SEQUENCE_LEN,
                                       start_offset=START_OFFSET, step=STEP,
                                       image_size=IMAGE_SIZE)
    print(f'Extracted frames shape: {images.shape}')

    # plot first 10 frames
    plot_images(images, frame_ids, 'Guess my activity')
