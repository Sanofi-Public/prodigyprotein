from prodigyprotein import TopPDirectedEvolutionSampler
import pytest
import tensorflow as tf


@pytest.fixture
def fixed_probabilities():
    """
    Fixture for creating fixed probabilities for testing.

    This function creates two fixed probability tensors. The first tensor has a higher probability for the 9th index,
    and the second tensor has a higher probability for the last two tokens. These tensors can be used for testing
    functions that require probability inputs.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        The two fixed probability tensors.

    Notes
    -----
    The first tensor is created by concatenating a tensor filled with the value 0.01 for the first 8 indices, a tensor
    filled with the value 0.90 for the 9th index, and a tensor filled with the value 0.09 for the last index. The
    second tensor is created by concatenating a tensor filled with the value 0.01 for the first 8 tokens, a tensor
    filled with the value 0.90 for the 9th token, and a tensor filled with the value 0.09 for the last token.
    """

    probabilities_index_9 = tf.concat(
        [
            tf.fill((2, 8, 1), value=0.01),
            tf.fill((2, 1, 1), value=0.90),
            tf.fill((2, 1, 1), value=0.09),
        ],
        axis=-2,
    )

    probabilities_token_0 = tf.concat(
        [
            tf.fill((2, 10, 8), value=0.01),
            tf.fill((2, 10, 1), value=0.90),
            tf.fill((2, 10, 1), value=0.09),
        ],
        axis=-1,
    )

    return probabilities_index_9, probabilities_token_0


def test_topp_directed_evolution_sampler(fixed_probabilities):
    """
    Test the functionality of the TopPDirectedEvolutionSampler class.

    This function tests the get_next_mutation method of the TopPDirectedEvolutionSampler class. It first creates an
    instance of the class with p=0.90. It then tests the get_next_mutation method by providing random probabilities
    and checking that the shapes and maximum values of the returned index and token match the expected values. It also
    tests the get_next_mutation method with fixed probabilities where the highest probabilities are for the 9th and
    8th indices and the 9th and 8th tokens, and checks that the returned index and token match these values.

    Parameters
    ----------
    fixed_probabilities : tuple
        The fixed probabilities for testing.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the shapes or maximum values of the returned index and token do not match the expected values, or if the
        returned index and token do not match the expected values for the fixed probabilities.

    Notes
    -----
    The TopPDirectedEvolutionSampler class is a class that samples mutations for a protein sequence based on
    probabilities. The get_next_mutation method returns the index and token of the next mutation based on the top-p
    provided probabilities.
    """
    sampler = TopPDirectedEvolutionSampler(p=0.90)

    # Test get_next_mutation
    probabilities = tf.random.uniform((10, 20, 30), minval=0, maxval=1)
    index, token = sampler._get_next_mutation(probabilities)
    assert index.shape == (10,)
    assert token.shape == (10,)
    assert tf.reduce_max(index) < 20
    assert tf.reduce_max(token) < 30

    probabilities_index_9, probabilities_token_0 = fixed_probabilities

    index, token = sampler._get_next_mutation(probabilities_index_9)

    assert tf.reduce_all(tf.logical_or(index == 9, index == 8))

    index, token = sampler._get_next_mutation(probabilities_token_0)

    assert tf.reduce_all(tf.logical_or(token == 9, token == 8))
