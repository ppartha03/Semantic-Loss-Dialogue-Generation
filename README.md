# Semantic-Loss-Dialogue-Generation

Splitting a single dialogue task into multiple tasks with dropping different output tokens to learn better encoder representation.

# Objective:

A machine intelligence task is defined as an objective to learn a mapping between a set of inputs and a set of outputs. T: I -> O

Although the algorithms trained to predict the functional mapping are probabilistic, the deterministic mapping learnt by the algorithms often fare well.

A training procedure that enables a model to learn such a mapping results in models that are not robust. Such a lack of generalization is often witnessed when the outputs are sampled at random with respect to a probability distribution.

An hypothesis to be validated is that the dynamic perturbation of outputs for the model to learn a slowly shifting target may be better than learning a fixed target.

To that end, we propose the output regularization by way of doing Meta-Learning for Dialogue. We dynamically generate new tasks by dropping out a few output tokens for the same input provided at the encoder. The model should adapt its hidden representation to learn a better discriminative representation of the encoder.

Although the concept is general, this is more relevant in NLP setting where the number of possible outputs is large. We hence test our hypothesis on dialogue datasets trained on Seq2Seq LSTM models. 

```

@inproceedings{
  parthasarathi2021Semantic,
  author       = {Parthasarathi, Prasanna and Abdelsalam, Mohamed and Chandar, Sarath and Pineau, Joelle},
  title        = {A Brief Study on the Effects of Training Generative Dialogue Models with a Semantic loss},
  year         = {2021},
  booktitle    = {Proceedings of the 22nd Annual SIGdial Meeting on Discourse and Dialogue},
  publisher    = {Association for Computational Linguistics},
}
