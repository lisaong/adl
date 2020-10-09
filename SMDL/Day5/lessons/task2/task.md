## Transformer

A Transformer consists of the following components:
- Positional Embedding:
  - This is a way to encode sequence information without using a recurrent cell. We essentially pass the position indices (e.g. `[0, 1, 2, ...]`) as an extra input. Think of this like a cross-sectional approach where we pass in `[(0, token1), (1, token2), ....]` instead of `[token1, token 2, ...]`.
  - Conceptually, once we have a position encoding (as a vector), we can compute similarity, compare relative positions of a token in different samples, etc.
- Multi-headed Self-attention:
  - Multi-headed means multiple sources of sequences are accepted, rather than 1 sequence.
  - Self-attention means the query-key and value are all on the **same** sequence (hence "self").
  
  ![famous picture](https://www.tensorflow.org/images/tutorials/transformer/multi_head_attention.png)

References:
- https://keras.io/examples/nlp/text_classification_with_transformer/
- https://www.tensorflow.org/tutorials/text/transformer
