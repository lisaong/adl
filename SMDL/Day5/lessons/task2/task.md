## Transformer

A Transformer consists of the following components:
- Token & Position Embedding:
- Multi-headed Self-attention

### 1. Token & Position Embedding

This is a way to embed sequence information without using a recurrent cell. We essentially pass the position indices (e.g. `[0, 1, 2, ...]`) as an extra input. Think of this like a cross-sectional approach where we pass in `[(0, token1), (1, token2), ....]` instead of `[token1, token 2, ...]`.

An Embedding layer produces the encoding into vectors. It is common to initialize the Embedding layer with a sinusoidal encoding. More details are described in [positional_encoding.md](positional_encoding.md)

The position embedding is **added** to the token embedding, to create the combined (position, token) encoding. You can think of it as different "biases" to the token encoding, to indicate where the token is located in the sequence.

![token_n_position_embedding](token_n_position_embedding.png)

### 2. Multi-headed Self-attention:

**Self-attention** means the query-key and value are all on the **same** sequence (hence "self"). It allows learning of contextual information (attention) on the source sequence based on the words in the source sequence. In comparison, Vanilla Attention uses a target (i.e. a different) sequence to learn contextual information on the source sequence.

**Multi-headed** means multiple parallel self-attention blocks are used. Analogous to having more neurons in Dense layer or channels in a Convolution layer to train more weights in parallel.
  
![internals](multiheaded_self_attention.png)

## Putting it Together

This is the architecture described in the famous [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

Note that while the paper describes an Encoder-Decoder model, Transformers can be used in non-encoder-decoder models as well, such as [text classification](https://keras.io/examples/nlp/text_classification_with_transformer/). They are essentially a replacement for LSTMs and GRUs.

### (1/2): Transformer Encoder

`transformer.py` demonstrates the encoder portion of the Transformer model. The encoder can be used as a feature extractor for a sequence learning task, effectively as a drop-in replacement for RNNs.

The Encoder architecture is depicted in the left-hand side of this model. Typically, this is a 6-layer stack (Decoder is described in the next section).
![transformer](https://www.tensorflow.org/images/tutorials/transformer/transformer.png)

### (2/2): Transformer Decoder

The Decoder is made up of similar building blocks as the encoder. It has a 6-layer Transformer Block with two multi-headed self-attention layers.
- Positional-encoding is applied to the current target sequence, then fed into 1 layer of  multi-headed self-attention.
- The output from the first layer is combined with the encoder output, into the 2nd multi-headed self-attention layer, etc.
- Finally, the output from the Decoder Transformer Block is passed through an MLP classifier to predict the next target token.


### References
- https://keras.io/examples/nlp/text_classification_with_transformer/: Basic Transformer block
- https://www.tensorflow.org/tutorials/text/transformer: Application of Transformer in Encoder/Decoder.
- https://link.medium.com/yA9Efs3iGab: Transformer - A Quick Run Through
- https://link.medium.com/9C0kSKRiGab: ELMO Embeddings from Language MOdels

