import tensorflow as tf
from .directed_evolution_sampler import DirectedEvolutionSampler


class TopKDirectedEvolutionSampler(DirectedEvolutionSampler):
    """
    A class used to represent a Top-K Directed Evolution Sampler.

    This class is a subclass of DirectedEvolutionSampler and implements top-k search.
    Only the top k tokens are considered per round of mutation. Amongst these top k, a token/residue is randomly selected from the probability distribution.

    ...

    Attributes
    ----------
    k : int
        The number of top tokens to consider per round of mutation.
    seed : int
        A seed for the random number generator.

    Methods
    -------
    _get_next_mutation(probabilities):
        Samples the next mutation from the given probability distribution considering only the top k tokens.

    get_config():
        Returns the configuration of the TopKDirectedEvolutionSampler, including the seed and k.
    """

    def __init__(
        self,
        k=3,
        seed=0,
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the TopKDirectedEvolutionSampler object.

        Parameters
        ----------
            k : int
                The number of top tokens to consider per round of mutation.
            seed : int
                A seed for the random number generator.
            **kwargs : dict
                Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed

    def _get_next_mutation(self, probabilities):
        """
        Samples the next mutation from the given probability distribution considering only the top k tokens.

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

        # Record dims
        batch_size, _, vocab_size = tf.shape(probabilities)

        # Flatten to vectors
        probabilities = tf.reshape(probabilities, (batch_size, -1))

        # Get top k per seq
        top_k_pred, top_k_indices = tf.math.top_k(probabilities, k=self.k, sorted=False)

        # Sample top k ind
        chosen_indices = tf.random.categorical(
            tf.math.log(top_k_pred), 1, seed=self.seed, dtype="int32"
        )
        # Generate ind to collect
        gather_ind = tf.concat([tf.range(batch_size)[:, None], chosen_indices], axis=-1)
        # Extract ind from vectors
        position_and_token = tf.gather_nd(top_k_indices, gather_ind)

        # Convert ind from vector to pos and token
        index = tf.math.ceil((position_and_token + 1) / vocab_size) - 1
        index = tf.cast(index, "int32")
        token = tf.math.floormod(position_and_token, vocab_size)

        return index, token

    def get_config(self):
        """
        Returns the configuration of the TopKDirectedEvolutionSampler, including the seed and k.

        Returns
        -------
            config : dict
                The configuration of the TopKDirectedEvolutionSampler.
        """
        config = super.get_config()
        config.update({"k": self.k, "seed": self.seed})
        return config
