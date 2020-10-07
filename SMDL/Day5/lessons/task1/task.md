## Attention Layer

An [Attention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention) layer can be used to
learn which inputs are "most relevant" in the decisions made by an artificial neural network.

Attention layers were introduced in 2015 (for Neural Machine Translation), but the general principle can apply to sequence-to-sequence models. 

Attention (a variant called multi-headed attention) is also the main element in Transformers. Transformers are tipped to be the successors to RNNs for sequential learning. We will cover Transformers in the next task.

The most basic Attention layer receives input in the form of query and key-value pairs:

|Application|Query Sequence|Key (maybe not sequence)|Value Sequence|
|---|---|---|---|
|Neural Machine Translation|target sequence|source sequence|source sequence|
|Recommendation System|target items|user profile|user purchase history|

* Notice that the query is referring to the **target**. This is because in Attention, we are trying to learn the relevance of sequential inputs (represented by key-value) on the sequential prediction (represented by the query). 

* Attention is just a generic framework to learn dynamic scores on sequential inputs, so the concept of keys are included to allow for extraneous (possibly non-sequential) descriptive information on the inputs. In most cases, where there is no extraneous information about the sequential inputs, then key == value (the case for Neural Machine Translation).

* The "relevance" is computed as a **weighted sum of the key and query sequence**. This yields a set of dynamic scores for each step in the value sequence v[t] (remember, value sequence is the **source** sequence).

* The dynamic scores are then **applied to the value sequence using another weighted sum**. This yields a set of dynamically weighted values, where higher weightage at v[t] means more relevance (more "attention" is given). This is called the **context vector** and is concatenated with the input to the subsequent layers.

Reference:
* Neural Machine Translation with attention: https://www.tensorflow.org/tutorials/text/nmt_with_attention

Further Enhancements:
* Self-attention: where key, value, and query 
* Global vs. Local attention: https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
* Spatial vs. Temporal attention: https://www.groundai.com/project/where-and-when-to-look-spatio-temporal-attention-for-action-recognition-in-videos/1
