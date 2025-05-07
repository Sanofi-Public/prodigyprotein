import tensorflow as tf
from .directed_evolution_sampler import DirectedEvolutionSampler


class TopPDirectedEvolutionSampler(DirectedEvolutionSampler):
    """
    A class used to represent a Top-P Directed Evolution Sampler.

    This class is a subclass of DirectedEvolutionSampler and implements top-p search.
    Only the tokens with cumulative probability less than or equal to p are considered per round of mutation. Amongst these tokens, a token/residue is randomly selected from the probability distribution.

    ...

    Attributes
    ----------
    p : float
        The cumulative probability threshold for considering tokens per round of mutation.
    seed : int
        A seed for the random number generator.

    Methods
    -------
    _get_next_mutation(probabilities):
        Samples the next mutation from the given probability distribution considering only the tokens with cumulative probability less than or equal to p.

    get_config():
        Returns the configuration of the TopPDirectedEvolutionSampler, including the seed and p.
    """

    def __init__(
        self,
        p=0.1,
        seed=0,
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the TopPDirectedEvolutionSampler object.

        Parameters
        ----------
            p : float
                The cumulative probability threshold for considering tokens per round of mutation.
            seed : int
                A seed for the random number generator.
            **kwargs : dict
                Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.p = p
        self.seed = seed

    def _get_next_mutation(self, probabilities):
        """
        Samples the next mutation from the given probability distribution considering only the tokens with cumulative probability less than or equal to p.

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
        batch_size, seq_len, vocab_size = tf.shape(probabilities)

        # Flatten to vector
        probabilities = tf.reshape(probabilities, (batch_size, -1))

        # Select top k across all tokens and positions
        sorted_k_pred, sorted_k_indices = tf.math.top_k(
            probabilities, k=tf.shape(probabilities)[1], sorted=True
        )

        # Compute cumulative probability
        cumul_probability = tf.cumsum(sorted_k_pred, axis=-1)

        # Mask all values exceed p
        mask = cumul_probability <= self.p

        # shift mask one position, otherwise if the first token
        # is above p all probability become 0!
        mask = tf.concat([tf.ones_like(mask[:, :1]), mask[:, :-1]], axis=-1)

        # Replace all probabilities where cumulative probability exceeded p
        probabilities = tf.where(
            condition=mask,
            x=sorted_k_pred,
            y=tf.zeros(shape=tf.shape(sorted_k_pred), dtype=sorted_k_pred.dtype),
        )

        chosen_indices = tf.random.categorical(
            tf.math.log(probabilities), 1, seed=self.seed, dtype="int32"
        )

        # Generate ind to collect
        gather_ind = tf.concat([tf.range(batch_size)[:, None], chosen_indices], axis=-1)

        # Extract ind from vectors
        position_and_token = tf.gather_nd(sorted_k_indices, gather_ind)

        # Convert ind from vector to pos and token
        index = tf.math.ceil((position_and_token + 1) / vocab_size) - 1
        index = tf.cast(index, "int32")
        token = tf.math.floormod(position_and_token, vocab_size)

        return index, token

    def get_config(self):
        """
        Returns the configuration of the TopPDirectedEvolutionSampler, including the seed and p.

        Returns
        -------
            config : dict
                The configuration of the TopPDirectedEvolutionSampler.
        """
        config = super.get_config()
        config.update({"p": self.p, "seed": self.seed})
        return config
