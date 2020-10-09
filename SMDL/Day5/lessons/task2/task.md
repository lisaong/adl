## Transformer

A Transformer consists of the following components:
- Positional Embedding:
  - This is a way to encode sequence information without using a recurrent cell. We essentially pass the position indices (e.g. `[0, 1, 2, ...]`) as an extra input. Think of this like a cross-sectional approach where we pass in `[(0, token1), (1, token2), ....]` instead of `[token1, token 2, ...]`.
  - An Embedding layer produces the encoding into vectors.
  - Conceptually, once we have a position encoding (as a vector), we can compute similarity, compare relative positions of a token in different samples, etc.
  - The position encoding is **added** to the token encoding, to create the combined (position, token) encoding. You can think of it as different "biases" to the token encoding, to indicate where the token is located in the sequence.
- Multi-headed Self-attention:
  - Self-attention means the query-key and value are all on the **same** sequence (hence "self"). It allows learning of contextual information (attention) on the source sequence based on the words in the source sequence. In comparison, Vanilla Attention uses a target (i.e. a different) sequence to learn contextual information on the source sequence.
  - Multi-headed means multiple parallel self-attention blocks are used. Analogous to having more neurons in Dense layer or channels in a Convolution layer to train more weights in parallel.
  
  ![famous picture](https://www.tensorflow.org/images/tutorials/transformer/multi_head_attention.png)

References:
- https://keras.io/examples/nlp/text_classification_with_transformer/
- https://www.tensorflow.org/tutorials/text/transformer
