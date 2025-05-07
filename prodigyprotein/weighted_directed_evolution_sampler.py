import tensorflow as tf
from .directed_evolution_sampler import DirectedEvolutionSampler


class WeightedDirectedEvolutionSampler(DirectedEvolutionSampler):
    """
    A class used to represent a Weighted Directed Evolution Sampler.

    This class is a subclass of DirectedEvolutionSampler and is used to sample the next mutation from a given probability distribution.
    The sampling is done randomly, but can be seeded for reproducibility.

    Attributes
    ----------
    seed : int, optional
        A seed for the random number generator, by default None

    Methods
    -------
    _get_next_mutation(probabilities):
        Samples the next mutation from the given probability distribution. It reshapes the probabilities and uses the TensorFlow random.categorical function to select a mutation.

    get_config():
        Returns the configuration of the WeightedDirectedEvolutionSampler, including the seed.
    """

    def __init__(self, seed=None, **kwargs):
        """
        Constructs all the necessary attributes for the WeightedDirectedEvolutionSampler object.

        Parameters
        ----------
            seed : int, optional
                A seed for the random number generator, by default None
            **kwargs : dict
                Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.seed = seed

    def _get_next_mutation(self, probabilities):
        """
        Samples the next mutation from the given probability distribution.

        Parameters
        ----------
            probabilities : Tensor
                The probabilities of the mutations.

        Returns
        -------
            index : Tensor
                The index of the mutation.
            token : Tensor
                The token of the mutation.
        """
        # Sample the next mutation from the probability distribution.
        batch_size = tf.shape(probabilities)[0]
        vocab_size = tf.shape(probabilities)[2]
        probabilities = tf.reshape(probabilities, (batch_size, -1))

        position_and_token = tf.random.categorical(
            tf.math.log(probabilities), 1, seed=self.seed, dtype="int32"
        )

        position_and_token = tf.reshape(position_and_token, (batch_size,))

        index = tf.math.ceil((position_and_token + 1) / vocab_size) - 1
        index = tf.cast(index, tf.int32)
        token = tf.math.floormod(position_and_token, vocab_size)

        return index, token

    def get_config(self):
        """
        Returns the configuration of the WeightedDirectedEvolutionSampler, including the seed.

        Returns
        -------
            config : dict
                The configuration of the WeightedDirectedEvolutionSampler.
        """
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
            }
        )
        return config
