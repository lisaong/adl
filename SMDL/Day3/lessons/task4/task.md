## Part 4: Training

The training process is as follows:

1. Pass the input through the encoder which return encoder output and the encoder hidden state.
2. The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.
3. The decoder returns the predictions and the decoder hidden state.
4. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
5. Use teacher forcing to decide the next input to the decoder. Teacher forcing is the technique where the target word is passed as the next input to the decoder.
6. The final step is to calculate the gradients and apply it to the optimizer and do backpropagation.

[Reference](https://www.tensorflow.org/tutorials/text/nmt_with_attention)