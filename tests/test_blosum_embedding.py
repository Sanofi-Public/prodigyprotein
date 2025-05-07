import pytest
import tensorflow as tf
from prodigyprotein import BlosumProbabilityEmbedding


@pytest.fixture
def blosum_embedding():
    cluster_pc = 80
    reorder_vocab = tf.constant(range(20), dtype=tf.int32)
    return BlosumProbabilityEmbedding(
        cluster_pc=cluster_pc, reorder_vocab=reorder_vocab
    )


@pytest.fixture
def blosum_embedding_reverse():
    cluster_pc = 80
    reorder_vocab = tf.constant(range(19, -1, -1), dtype=tf.int32)
    return BlosumProbabilityEmbedding(
        cluster_pc=cluster_pc, reorder_vocab=reorder_vocab
    )


def test_blosum_probability_embedding():
    """
    Test the functionality of the BlosumProbabilityEmbedding layer.

    This function tests the build and call methods of the BlosumProbabilityEmbedding layer. It first creates an
    instance of the layer with a specified input dimension. It then tests the build method by providing an input shape
    and checking that the shape of the blosum_probabilities attribute matches the expected shape. It also tests the
    call method by providing input data and checking that the shape of the output data matches the expected shape.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the shape of the blosum_probabilities attribute or the output data does not match the expected shape.

    Notes
    -----
    The BlosumProbabilityEmbedding layer is a custom layer that embeds the input data using BLOSUM probabilities. The
    build method initialises the blosum_probabilities attribute based on the input shape, and the call method applies
    returns the blosum probabilities for the given residue.
    """
    layer = BlosumProbabilityEmbedding(45)

    # Test build
    input_shape = (10, 20)
    layer.build(input_shape)
    assert layer.blosum_probabilities.shape == (20, 20)

    # Test call
    input_data = tf.ones((10, 20), dtype=tf.int32)
    output_data = layer.call(input_data)

    assert output_data.shape == (10, 20, 20)


def test_init():
    with pytest.raises(ValueError):
        BlosumProbabilityEmbedding(cluster_pc=None)

    with pytest.raises(ValueError):
        BlosumProbabilityEmbedding(
            cluster_pc=80, reorder_vocab=tf.constant(range(10), dtype=tf.int32)
        )

    with pytest.raises(ValueError):
        BlosumProbabilityEmbedding(
            cluster_pc=80,
            reorder_vocab=tf.reshape(
                tensor=tf.constant(range(20), dtype=tf.int32), shape=(4, 5)
            ),
        )


def test_get_config(blosum_embedding):
    config = blosum_embedding.get_config()
    assert config["cluster_pc"] == 80


def test_from_config(blosum_embedding):
    config = blosum_embedding.get_config()
    new_layer = BlosumProbabilityEmbedding.from_config(config)
    assert new_layer.cluster_pc == 80


def test_build(blosum_embedding):
    input_shape = tf.TensorShape([10])
    blosum_embedding.build(input_shape)
    assert hasattr(blosum_embedding, "blosum_probabilities")
    assert blosum_embedding.blosum_probabilities.shape == [20, 20]


def test_call(blosum_embedding):
    inputs = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.int32)
    blosum_embedding.build(inputs.shape)
    outputs = blosum_embedding.call(inputs)
    assert outputs.shape == (10, 20)


def test_reorder(blosum_embedding, blosum_embedding_reverse):
    inputs = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.int32)
    blosum_embedding.build(inputs.shape)
    blosum_embedding_reverse.build(inputs.shape)

    reversed_embeddings = tf.reverse(
        blosum_embedding_reverse.blosum_probabilities, axis=[0, 1]
    )

    assert tf.reduce_all(reversed_embeddings == blosum_embedding.blosum_probabilities)
