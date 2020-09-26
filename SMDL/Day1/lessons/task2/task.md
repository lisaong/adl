## Labels to Categorical
String labels need to be converted to output columns for training in Tensorflow-Keras.

There are two steps to this process. Step 2 is not needed for binary classification:
1. First, we take each string label and convert it to an ordinal number. For example, `['Happy', 'Sad', 'Contented']` can be represented by `[0, 1, 2]`. We use the `LabelEncoder` from scikit-learn to find the unique string labels and assign a number to each. To convert-back to strings, the `LabelEncoder.inverse_transform()` method is available. 
2. Next, if we have more than 2 classes, an extra step is needed to convert the ordinal (numeric) labels into columns, one column per label. For example `[0, 1, 2]` will become 
    ```
    [[1, 0, 0],
     [0, 1, 0], 
     [0, 0, 1]
    ```
     We can use the Tensorflow-Keras `to_categorical` function to do the conversion. To convert back to ordinal (numeric), we typically use `ndarray.argmax(axis=1)`.

[Label Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

[to_categorical](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical)
