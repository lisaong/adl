## Video Frame Dataset

This demo shows how to download, extract, and create a video frame dataset from zip files.

1. Use [tf.keras.utils.get_file](https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file) to download the dataset from URLs. This will download the dataset, unzip it, and cache under the `~/.keras/datasets` folder.

2. Use [tf.data.Dataset.list_files](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#list_files) to glob and find videos in the dataset folder.

3. Use [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) to extract frames from each video into a numpy array, with a label.

[Reference](https://www.tensorflow.org/guide/data#decoding_image_data_and_resizing_it)