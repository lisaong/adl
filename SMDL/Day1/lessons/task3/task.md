## Word Embeddings

When processing text, word embeddings are often used to represent large vocabularies with a compressed vector.

For example, if we have 100 unique words in our vocabulary:
- No embedding: 100 features per word per document. 
- With embedding (of output dimension 2): 2 features per word per document

```
Corpus:
  ['we the citizens of singapore',
  'pledge ourselves as one united people']
  
  Vocabulary size = 11 (number of unique words)

No Embedding:
  'we the citizens of singapore' => 5 words * 11 features per word

  'pledge ourselves as one united people' => 6 words * 11 features per word

With Embedding (output_dim 2):

  'We the citizens of Singapore' => 5 words * 2 features per word

  'pledge ourselves as one united people' => 6 words * 2 features per word

```

The output dimension is proportional to the vocabulary size, but is usually something very small compared to the number of unique words. For example, [GloVe](https://nlp.stanford.edu/projects/glove/) uses only 50 dimensions to represent a vocabulary of 400,000+ words.

### Word Vectors
The learnt embedding layer will also provide word vectors, which can be said to represent the contextual meaning of each word. The context is defined by how the word is used in the corpus.

![vectors](vectors.png)

Words that are similar in contextual meaning will have vectors closer together (e.g. 'achieve' and 'progress).

For larger dimensional word vectors, you can use the [Embedding Projector] (https://projector.tensorflow.org/) to explore the word vector space.
