from prodigyprotein import (
    DirectedEvolutionSampler,
)
import pytest
import tensorflow as tf


def test_mask_intervals():
    """
    Test the _mask_intervals method of the DirectedEvolutionSampler class.

    This function tests the _mask_intervals method by providing a 3D tensor and a list of intervals. It checks that
    the returned tensor has the expected shape and that the masked columns contain -inf.

    Raises
    ------
    AssertionError
        If the shape of the returned tensor does not match the expected shape, or if the masked columns do not contain
        -inf.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Create a 3D tensor
    tensor_3d = tf.random.uniform((2, 10, 3), minval=0, maxval=1, dtype=tf.float64)

    # Define intervals
    intervals = [(2, 4), (6, 8)]

    # Call the _mask_intervals method
    masked_tensor = sampler._mask_intervals(tensor_3d, intervals)

    # Check that the shape of the returned tensor matches the expected shape
    assert masked_tensor.shape == tensor_3d.shape

    # Check that the masked columns contain -inf
    for i in range(tensor_3d.shape[1]):
        if not any(start <= i < end for start, end in intervals):
            assert tf.reduce_all(tf.math.is_inf(masked_tensor[:, i, :]))


def test_validate_intervals():
    """
    Test the _validate_intervals method of the DirectedEvolutionSampler class.

    This function tests the _validate_intervals method by providing a list of intervals. It checks that
    the returned intervals are valid and correctly formatted.

    Raises
    ------
    AssertionError
        If the returned intervals do not match the expected intervals.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define intervals
    intervals = [(2, 4), (6, 8)]

    # Call the _validate_intervals method
    validated_intervals = sampler._validate_intervals(intervals)

    # Check that the returned intervals match the expected intervals
    assert validated_intervals == intervals


def test_validate_intervals_invalid_input():
    """
    Test the _validate_intervals method of the DirectedEvolutionSampler class with invalid input.

    This function tests the _validate_intervals method by providing invalid input and checks that
    it raises the appropriate exceptions.

    Raises
    ------
    ValueError
        If the input is not a list, the elements of the list are not tuples, the tuples do not contain exactly two
        elements, the elements of the tuples are not integers, the end index is not greater than the start index,
        the start positions are not in order, the end positions are not in order, or the intervals overlap.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define invalid inputs
    invalid_inputs = [
        "not a list",  # Input is not a list
        [(2, 4), "not a tuple"],  # Elements of the list are not tuples
        [(2, 4), (6,)],  # Tuples do not contain exactly two elements
        [(2, 4), (6, "not an integer")],  # Elements of the tuples are not integers
        [(2, 4), (4, 3)],  # End index is not greater than start index
        [(2, 4), (1, 3)],  # Start positions are not in order
        [(2, 4), (3, 5)],  # End positions are not in order
        [(2, 4), (3, 5)],  # Intervals overlap
    ]

    # Call the _validate_intervals method with invalid input and check that it raises the appropriate exceptions
    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError):
            sampler._validate_intervals(invalid_input)


def test_skip_mutation():
    """
    Test the skip_mutation method of the DirectedEvolutionSampler class.

    This function tests the skip_mutation method by providing a protein sequence, probabilities, index, and next_mutation.
    It checks that the returned index and next_mutation are correctly updated.

    Raises
    ------
    AssertionError
        If the returned index and next_mutation do not match the expected values.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define protein sequence, probabilities, index, and next_mutation
    protein_sequence = tf.constant([[1, 2, 3, 4, 5]])
    probabilities = tf.constant([[[0.0, 0.0, 0.0, 0.0, 0.0]]])
    index = tf.constant([4])
    next_mutation = tf.constant([6])

    # Call the skip_mutation method
    updated_index, updated_next_mutation = sampler._skip_mutation(
        probabilities, protein_sequence, index, next_mutation
    )

    # Check that the returned index and next_mutation match the expected values
    print(updated_index)
    assert tf.reduce_all(updated_index == tf.constant([0]))
    assert tf.reduce_all(updated_next_mutation == tf.constant([1]))

    # Define protein sequence, probabilities, index, and next_mutation (should leave index and next_mutation unchanged)
    protein_sequence = tf.constant([[1, 2, 3, 4, 5]])
    probabilities = tf.constant([[[0.0, 0.0, 0.0, 0.0, 0.5]]])
    index = tf.constant([4])
    next_mutation = tf.constant([0])

    # Call the skip_mutation method
    updated_index, updated_next_mutation = sampler._skip_mutation(
        probabilities, protein_sequence, index, next_mutation
    )

    # Check that the returned index and next_mutation match the expected values
    print(updated_index)
    assert tf.reduce_all(updated_index == tf.constant([4]))
    assert tf.reduce_all(updated_next_mutation == tf.constant([0]))


def test_skip_mutation_invalid_input():
    """
    Test the skip_mutation method of the DirectedEvolutionSampler class with invalid input.

    This function tests the skip_mutation method by providing invalid input and checks that
    it raises the appropriate exceptions.

    Raises
    ------
    ValueError
        If the dimensions of input tensors (index and next_mutation) are not correct.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define invalid inputs
    invalid_inputs = [
        (
            tf.constant([[1, 2, 3, 4, 5]]),
            tf.constant([[0.1, 0.2, 0.3, 0.4, 0.0]]),
            tf.constant([[4]]),
            tf.constant([6]),
        ),  # Index rank is not 1
        (
            tf.constant([[1, 2, 3, 4, 5]]),
            tf.constant([[0.1, 0.2, 0.3, 0.4, 0.0]]),
            tf.constant([4]),
            tf.constant([[6]]),
        ),  # Next_mutation rank is not 1
    ]

    # Call the skip_mutation method with invalid input and check that it raises the appropriate exceptions
    for protein_sequence, probabilities, index, next_mutation in invalid_inputs:
        with pytest.raises(ValueError):
            sampler._skip_mutation(
                probabilities, protein_sequence, index, next_mutation
            )


def test_geometric_mean():
    """
    Test the geometric_mean method of the DirectedEvolutionSampler class.

    This function tests the geometric_mean method by providing two probabilities. It checks that
    the returned geometric mean matches the expected value.

    Raises
    ------
    AssertionError
        If the returned geometric mean does not match the expected value.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define probabilities
    prob_a = tf.constant([0.5])
    prob_b = tf.constant([0.5])

    # Call the geometric_mean method
    geo_mean = sampler._geometric_mean(prob_a, prob_b)

    # Check that the returned geometric mean matches the expected value
    assert tf.reduce_all(tf.abs(geo_mean - tf.constant([0.5], dtype="float64")) < 1e-6)


def test_geometric_mean_zero_probability():
    """
    Test the geometric_mean method of the DirectedEvolutionSampler class with zero probability.

    This function tests the geometric_mean method by providing a zero probability. It checks that
    the returned geometric mean is zero.

    Raises
    ------
    AssertionError
        If the returned geometric mean is not zero.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define probabilities
    prob_a = tf.constant([0.0])
    prob_b = tf.constant([0.5])

    # Call the geometric_mean method
    geo_mean = sampler._geometric_mean(prob_a, prob_b)

    # Check that the returned geometric mean is zero
    assert tf.reduce_all(geo_mean == tf.constant([0.0], dtype="float64"))


def test_compute_probabilities_zero_logits():
    """
    Test the compute_probabilities method of the DirectedEvolutionSampler class with zero logits.

    This function tests the compute_probabilities method by providing zero logits. It checks that
    the returned probabilities are equal.

    Raises
    ------
    AssertionError
        If the returned probabilities are not equal.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define zero logits
    logits = tf.constant([[[0.0], [0.0], [0.0]]])

    # Call the compute_probabilities method
    probabilities = sampler._compute_probabilities(logits)

    # Check that the returned probabilities are equal to one for each position in sequence
    assert tf.reduce_all(tf.reduce_sum(probabilities, axis=-1) == 1.0)


def test_devolution():
    """
    Test the devolution method of the DirectedEvolutionSampler class.

    This function tests the devolution method by providing mutational probabilities and a step number.
    It checks that the returned variants match the expected values.

    Raises
    ------
    AssertionError
        If the returned variants do not match the expected values.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define mutational probabilities
    mutational_probabilities = {
        "step_0": tf.constant([[[0.0, 0.2, 0.3, 0.3, 0.2]]]),
        "step_1": tf.constant([[[0.3, 0.2, 0.3, 0.0, 0.2]]]),
        "step_2": tf.constant([[[0.3, 0.2, 0.0, 0.3, 0.2]]]),
    }

    # Call the devolution method
    variants = sampler.devolution(mutational_probabilities, 1)

    # Check that the returned variants match the expected values
    assert tf.reduce_all(variants == tf.constant(tf.constant([[3]])))


def test_multiple_devolution():
    """
    Test the devolution method of the DirectedEvolutionSampler class.

    This function tests the devolution method by providing mutational probabilities and a step number.
    It checks that the returned variants match the expected values.

    Raises
    ------
    AssertionError
        If the returned variants do not match the expected values.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define mutational probabilities
    mutational_probabilities = {
        "step_0": tf.constant([[[0.0, 0.1, 0.3, 0.4, 0.1], [0.0, 0.1, 0.3, 0.4, 0.1]]]),
        "step_1": tf.constant([[[0.3, 0.2, 0.3, 0.0, 0.2], [0.3, 0.2, 0.3, 0.0, 0.2]]]),
        "step_2": tf.constant([[[0.3, 0.2, 0.0, 0.3, 0.2], [0.3, 0.2, 0.0, 0.3, 0.2]]]),
    }

    # Call the devolution method
    variants = sampler.devolution(mutational_probabilities, 1)

    # Check that the returned variants match the expected values
    assert tf.reduce_all(variants == tf.constant(tf.constant([[3], [3]])))


def test_devolution_invalid_step():
    """
    Test the devolution method of the DirectedEvolutionSampler class with an invalid step number.

    This function tests the devolution method by providing an invalid step number and checks that
    it raises the appropriate exceptions.

    Raises
    ------
    ValueError
        If the step number is greater than or equal to the number of directed evolution steps.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define mutational probabilities
    mutational_probabilities = {
        "step_0": tf.constant([[[0.0, 0.2, 0.3, 0.3, 0.2]]]),
        "step_1": tf.constant([[[0.3, 0.2, 0.3, 0.0, 0.2]]]),
        "step_2": tf.constant([[[0.3, 0.2, 0.0, 0.3, 0.2]]]),
    }

    # Call the devolution method with an invalid step number and check that it raises the appropriate exceptions
    with pytest.raises(ValueError):
        sampler.devolution(mutational_probabilities, 3)


def test_devolution_llr_threshold():
    """
    Test the devolution method of the DirectedEvolutionSampler class with llr_threshold set.

    This function tests the devolution method by setting the llr_threshold and checks that
    it raises the appropriate exceptions.

    Raises
    ------
    ValueError
        If the llr_threshold is set.
    """
    # Create an instance of the DirectedEvolutionSampler class with llr_threshold set
    sampler = DirectedEvolutionSampler(llr_threshold=0.5)

    # Define mutational probabilities
    mutational_probabilities = {
        "step_0": tf.constant([[[0.0, 0.2, 0.3, 0.3, 0.2]]]),
        "step_1": tf.constant([[[0.3, 0.2, 0.3, 0.0, 0.2]]]),
        "step_2": tf.constant([[[0.3, 0.2, 0.0, 0.3, 0.2]]]),
    }

    # Call the devolution method and check that it raises the appropriate exceptions
    with pytest.raises(ValueError):
        sampler.devolution(mutational_probabilities, 1)


def test_devolution_multiple_zeros():
    """
    Test the devolution method of the DirectedEvolutionSampler class with multiple zeros in probabilities.

    This function tests the devolution method by providing mutational probabilities with multiple zeros and checks that
    it raises the appropriate exceptions.

    Raises
    ------
    ValueError
        If there are multiple zeros in the probabilities for any position.
    """
    # Create an instance of the DirectedEvolutionSampler class
    sampler = DirectedEvolutionSampler()

    # Define mutational probabilities with multiple zeros
    mutational_probabilities = {
        "step_0": tf.constant([[[0.0, 0.0, 0.3, 0.3, 0.2]]]),
        "step_1": tf.constant([[[0.3, 0.2, 0.3, 0.0, 0.2]]]),
        "step_2": tf.constant([[[0.3, 0.2, 0.0, 0.3, 0.2]]]),
    }

    # Call the devolution method and check that it raises the appropriate exceptions
    with pytest.raises(ValueError):
        sampler.devolution(mutational_probabilities, 1)
