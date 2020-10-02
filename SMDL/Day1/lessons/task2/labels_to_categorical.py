# Labels to Categorical demonstration
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

my_labels = ['Happy', 'Sad', 'Puzzled', 'Sad', 'Happy', 'Surprised']

# first we label encode to convert to numbers
le = LabelEncoder()
my_labels_encoded = le.fit_transform(my_labels)

print('=======================')
print('string => labels')
print(my_labels_encoded)
print('Classes', le.classes_)

# then we create categorical versions (one-hot encoding)
y = to_categorical(my_labels_encoded)

print('=======================')
print('labels => categorical')
print(y)

# conversion back
y_labels = y.argmax(axis=1)

print('=======================')
print('Reverse: categorical => labels')
print(y_labels)

my_labels_decoded = le.inverse_transform(y_labels)

print('=======================')
print('Reverse: labels => string')
print(my_labels_decoded)
