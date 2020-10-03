## Video Frame Dataset

This demo shows how to download, extract, and create a video frame dataset from zip files.

1. Use [tf.keras.utils.get_file](https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file) to download the dataset from URLs. This will download the dataset, unzip it, and cache under the `~/.keras/datasets` folder.

2. Use [glob](https://docs.python.org/3/library/glob.html) to glob and find videos in the dataset folder.

3. Extract frames from each video into a numpy array, with a label, using the function developed in task 1 (`frame_extractor.py).

4. Stack the dataset and labels. The shape of X should be (batch, sequence, height, width, channels). The shape of y should be (batch, 1).

5. Save the datasets for the next task.