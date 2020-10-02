# Embeddings demonstration

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

my_text = np.array(['We, the citizens of Singapore',
                    'pledge ourselves as one united people',
                    'regardless of race, language or religion',
                    'to build a democratic society',
                    'based on justice and equality',
                    'so as to achieve happiness, prosperity',
                    'and progress for our nation.'])

# create some simple labels, we'll just label every text
# based on whether it contains a verb (action)
my_labels = np.array(['None',
                      'Action',  # "pledge"
                      'None',
                      'Action',  # "to build"
                      'None',
                      'Action',  # "to achieve"
                      'None'])

# Vectorizer
vectorizer = TextVectorization(max_tokens=100, output_sequence_length=5)
vectorizer.adapt(my_text)

X = vectorizer(my_text).numpy()

print('=======================')
print('Original Text', my_text)
print('Vectorized sequence', X)
print('=======================')
print('Vocabulary', vectorizer.get_vocabulary())

# Labels to Categorical
le = LabelEncoder()
y = le.fit_transform(my_labels)
# we don't need to_categorical because these are binary labels

print('=======================')
print('string => labels (binary)')
print(y)
print('Classes', le.classes_)

# Train a simple Neural Network with an Embedding layer
vocab_size = len(vectorizer.get_vocabulary())
word_vector_size = 2

model_input = Input(shape=(5,))
x = Embedding(input_dim=vocab_size, output_dim=word_vector_size)(model_input)
x = Flatten()(x)
x = Dense(8, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(model_input, x)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X, y, epochs=5, batch_size=2)  # we have too few data,
                                         # so no validation set in this demo

# let's examine the Embedding layer, we'll throw the model away
embedding_matrix = model.layers[1].get_weights()
vocab = vectorizer.get_vocabulary()

fig, ax = plt.subplots()
for word, vector in zip(vocab, embedding_matrix[0]):
    print('====')
    print(word)
    print(vector)

    # let's plot the word vectors
    ax.scatter(vector[0], vector[1])
    ax.text(vector[0], vector[1], word)
ax.set_title('2-dimensional word vectors learnt via training a classifier')
plt.savefig('vectors.png')
plt.show()
