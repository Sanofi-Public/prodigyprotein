import tensorflow as tf
from .directed_evolution_sampler import DirectedEvolutionSampler


class GreedyDirectedEvolutionSampler(DirectedEvolutionSampler):
    """
    A class used to represent a Greedy Directed Evolution Sampler.

    This class is a subclass of the DirectedEvolutionSampler class and overrides the get_next_mutation method.

    Methods
    -------
    _get_next_mutation(probabilities)
        Gets the next mutation based on the given probabilities.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the GreedyDirectedEvolutionSampler object.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments.
        """

        super().__init__(**kwargs)

    def _get_next_mutation(self, probabilities):
        """
        Gets the next mutation based on the given probabilities.

        Parameters
        ----------
        probabilities : tf.Tensor
            A 3D tensor of shape (batch_size, sequence_length, vocab_size) representing the probabilities of each token at each
            position for each sequence in the batch.

        Returns
        -------
        index : tf.Tensor
            A 1D tensor of shape (batch_size,) representing the position of the maximum probability for each sequence in the batch.
        token : tf.Tensor
            A 1D tensor of shape (batch_size,) representing the token of the maximum probability for each sequence in the batch.
        """
        batch_size, _, vocab_size = tf.shape(probabilities)
        probabilities = tf.reshape(probabilities, (batch_size, -1))

        position_and_token = tf.argmax(probabilities, axis=-1)
        position_and_token = tf.cast(position_and_token, "int32")
        position_and_token = tf.reshape(position_and_token, (batch_size,))

        index = tf.math.ceil((position_and_token + 1) / vocab_size) - 1
        index = tf.cast(index, "int32")
        token = tf.math.floormod(position_and_token, vocab_size)

        return index, token
