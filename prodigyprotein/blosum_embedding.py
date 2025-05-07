import tensorflow as tf
from .blosum_initialiser import BlosumInitializer


class BlosumProbabilityEmbedding(tf.keras.layers.Layer):
    """
    A class used to represent a Blosum Probability Embedding Layer.

    This class is a subclass of the tf.keras.layers.Layer class and overrides the __init__, get_config, build, call, and compute_output_shape methods.

    Methods
    -------
    get_config()
        Gets the configuration of the BlosumProbabilityEmbedding layer.
    build(inputs_shape)
        Builds the BlosumProbabilityEmbedding layer.
    call(inputs)
        Calls the BlosumProbabilityEmbedding layer on the given inputs.
    compute_output_shape(input_shape)
        Computes the output shape of the BlosumProbabilityEmbedding layer.
    """

    def __init__(
        self,
        cluster_pc,
        reorder_vocab=None,
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the BlosumProbabilityEmbedding object.

        Parameters
        ----------
        cluster_pc : int
            The percentage of clusters to consider.
        reorder_vocab : tf.Tensor, optional
            The reordered vocabulary tensor.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        if cluster_pc is None:
            raise ValueError("`cluster_pc` must be an Integer, received `None`.")

        if reorder_vocab is not None:
            shape = tf.shape(reorder_vocab)
            rank = len(shape)
            length = shape[0]

            if rank != 1:
                raise ValueError(
                    "reorder vocab must be rank 1, received rank {}".format(rank)
                )
            if length != 20:
                raise ValueError(
                    "reorder vocab must be length 20, received length {}".format(length)
                )

        self.initializer = BlosumInitializer(
            cluster_pc=cluster_pc, reorder_vocab=reorder_vocab
        )
        self.cluster_pc = cluster_pc
        self.reorder_vocab = reorder_vocab

    def get_config(self):
        """
        Gets the configuration of the BlosumProbabilityEmbedding layer.

        Returns
        -------
        config : dict
            The configuration of the BlosumProbabilityEmbedding layer.
        """
        config = super().get_config()
        config.update(
            {"cluster_pc": self.cluster_pc, "reorder_vocab": self.reorder_vocab}
        )
        return config

    @classmethod
    def from_config(cls, config):
        cluster_pc = config.pop("cluster_pc")
        reorder_vocab = config.pop("reorder_vocab")

        return cls(cluster_pc, reorder_vocab, **config)

    def build(self, inputs_shape):
        """
        Builds the BlosumProbabilityEmbedding layer.

        Parameters
        ----------
        inputs_shape : tf.TensorShape
            The shape of the inputs to the layer.
        """

        self.blosum_probabilities = self.add_weight(
            name="blosum_probabilties",
            shape=[20, 20],
            initializer=self.initializer,
            trainable=False,
        )

        self.built = True

    def call(self, inputs):
        """
        Calls the BlosumProbabilityEmbedding layer on the given inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            The inputs to the layer.

        Returns
        -------
        blosum_probability_embedding : tf.Tensor
            The Blosum probability embedding of the inputs.
        """
        blosum_probabilities = tf.convert_to_tensor(self.blosum_probabilities)

        blosum_probability_embedding = tf.gather(
            params=blosum_probabilities, indices=inputs, batch_dims=0
        )

        return blosum_probability_embedding

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the BlosumProbabilityEmbedding layer.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the inputs to the layer.

        Returns
        -------
        output_shape : tf.TensorShape
            The shape of the outputs of the layer.
        """
        output_shape = list(input_shape)
        output_shape.append(20)
        return tuple(output_shape)
