## Time Distributed CNN

1. Load the dataset from the previous task.

2. After train-test-split, use [tf.data.Dataset.from_tensor_slices](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices) to create 
a batched dataset for training. Training set is batched up and repeated for batches per epoch.

3. 
