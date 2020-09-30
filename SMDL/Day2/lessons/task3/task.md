## Returning Sequences

An Recurrent Neural Network (LSTM or GRU) can be used to generate sequences.

Using the `return_sequences=True` parameter, an LSTM or GRU can return the (intermediate) output activations of every time step.

Situations where this is useful:
- When the final output activation does not contain enough information for the next layers to learn.
- When stacking multiple LSTM or GRU layers, all layers except for the last LSTM/GRU will need to return sequences. This is because LSTM/GRUs expect inputs with the timesteps dimension (3D tensors).

Notes:
- `return_sequences` returns the output activations only. This is different from `return_state`, which returns the hidden states. Hidden states are used in Sequence-to-Sequence networks (see Day 3).
