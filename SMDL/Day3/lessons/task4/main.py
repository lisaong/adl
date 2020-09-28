# Toy RNN Encoder-Decoder: Part 4 - Training
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from ..task1.main import MyEncoder
from ..task2.main import MyDecoder
from ..task3.main import loss_function, loss_equation

# source text
english_text = ['Ask, and it will be given to you',
                'seek, and you will find',
                'knock, and it will be opened to you.']

# target text
spanish_text = ['Pidan, y se les dará',
                'busquen, y encontrarán',
                'llamen, y se les abrirá.']

BATCH_SIZE = 3
EMBEDDING_SIZE = 2
BOTTLENECK_UNITS = 1

START_TOKEN = '<start>'
END_TOKEN = '<end>'

# append start and end tokens, this will indicate when translation should start & stop
src_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text]

src_vectorizer = TextVectorization(output_sequence_length=10)
src_vectorizer.adapt(src_text)
src_sequences = src_vectorizer(src_text)
src_vocab_size = len(src_vectorizer.get_vocabulary())

target_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]

target_vectorizer = TextVectorization(output_sequence_length=10)
target_vectorizer.adapt(target_text)
target_sequences = target_vectorizer(target_text)
target_vocab_size = len(target_vectorizer.get_vocabulary())

if __name__ == '__main__':
    pass