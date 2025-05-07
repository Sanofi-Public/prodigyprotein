import tensorflow as tf
from prodigyprotein import WeightedDirectedEvolutionSampler
import pytest


@pytest.fixture
def sampler():
    seed = 42
    return WeightedDirectedEvolutionSampler(seed=seed)


class MockMLMAbsZero:
    def predict(self, inputs):
        if type(inputs) == tuple:
            batch_size, seq_len = tf.shape(inputs[0])

        else:

            # Mock prediction method to return fixed logits for testing
            batch_size, seq_len = tf.shape(inputs["input_ids"])

        zeros_logit = tf.zeros(shape=(batch_size, 1, 2), dtype=tf.float64)
        ones_logit = tf.ones(shape=(batch_size, seq_len - 1, 2), dtype=tf.float64)

        logits = tf.concat([zeros_logit, ones_logit], axis=1)
        return {"logits": logits}

    def __call__(self, inputs):

        if type(inputs) == tuple:
            batch_size, seq_len = tf.shape(inputs[0])

        else:

            # Mock prediction method to return fixed logits for testing
            batch_size, seq_len = tf.shape(inputs["input_ids"])

        zeros_logit = tf.zeros(shape=(batch_size, seq_len, 1), dtype=tf.float64)
        ones_logit = tf.ones(shape=(batch_size, seq_len, 1), dtype=tf.float64)

        logits = tf.concat([zeros_logit, ones_logit], axis=-1)
        return {"logits": logits}


class MockMLMInf:
    def predict(self, inputs):
        if type(inputs) == tuple:
            batch_size, seq_len = tf.shape(inputs[0])

        else:

            # Mock prediction method to return fixed logits for testing
            batch_size, seq_len = tf.shape(inputs["input_ids"])

        ones_logit = tf.ones(shape=(batch_size, seq_len, 1), dtype=tf.float64)
        inf_logit = tf.fill(value=tf.math.log(0.0), dims=(batch_size, seq_len, 1))
        inf_logit = tf.cast(inf_logit, tf.float64)
        logits = tf.concat([ones_logit, inf_logit], axis=-1)
        logits = tf.cast(logits, tf.float64)
        return {"logits": logits}

    def __call__(self, inputs):

        if type(inputs) == tuple:
            batch_size, seq_len = tf.shape(inputs[0])

        else:

            # Mock prediction method to return fixed logits for testing
            batch_size, seq_len = tf.shape(inputs["input_ids"])

        ones_logit = tf.ones(shape=(batch_size, seq_len, 1), dtype=tf.float64)
        inf_logit = tf.fill(value=tf.math.log(0.0), dims=(batch_size, seq_len, 1))
        inf_logit = tf.cast(inf_logit, tf.float64)
        logits = tf.concat([ones_logit, inf_logit], axis=-1)
        logits = tf.cast(logits, tf.float64)
        return {"logits": logits}


@pytest.fixture
def mock_mlm_inf():
    return MockMLMInf()


@pytest.fixture
def mock_mlm_zero():
    return MockMLMAbsZero()


def test_get_next_mutation(sampler):
    probabilities = tf.constant([[[0.1, 0.9], [0.8, 0.2]]], dtype=tf.float32)
    index, token = sampler._get_next_mutation(probabilities)

    assert index.shape == (1,)
    assert token.shape == (1,)
    assert tf.reduce_all(index >= 0)
    assert tf.reduce_all(token >= 0)


def test_zero_probability_positions_not_sampled(sampler):
    probabilities = tf.constant([[[0.0, 1.0], [0.0, 0.0]]], dtype=tf.float32)
    index, token = sampler._get_next_mutation(probabilities)

    assert index.shape == (1,)
    assert token.shape == (1,)
    assert tf.reduce_all(index == 0)
    assert tf.reduce_all(token == 1)


def test_zero_probability_positions_not_sampled_with_multiple_batches(sampler):
    probabilities = tf.constant(
        [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]], dtype=tf.float32
    )
    index, token = sampler._get_next_mutation(probabilities)

    assert index.shape == (2,)
    assert token.shape == (2,)
    assert tf.reduce_all(index == [0, 1])
    assert tf.reduce_all(token == [1, 1])


def test_get_config(sampler):
    config = sampler.get_config()
    assert config["seed"] == sampler.seed


def test_empty_probability_tensor(sampler):
    probabilities = tf.constant([[[]]], dtype=tf.float32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        sampler._get_next_mutation(probabilities)


def test_inf_ignored(mock_mlm_inf, sampler):
    # One tokenised sequence of length 2, shape: [1, 2]
    sequence = tf.constant([[0, 1]])
    attention_mask = tf.ones_like(sequence)

    input_sequence = {"input_ids": sequence, "attention_mask": attention_mask}

    variants, mut_prob, logits = sampler(
        sequence=input_sequence,
        max_steps=1,
        mask_token=2,
        masked_marginals=False,
        model=mock_mlm_inf,
    )

    assert variants.shape == (1, 2)
    assert tf.reduce_all(variants == tf.constant([[0, 0]], dtype=tf.int32))

    # Input tokens are automatically masked, this means 1. value for position 0 becomes 0
    assert tf.reduce_all(
        mut_prob["step_0"] == tf.constant([[[0.0, 0.0], [1.0, 0.0]]], dtype=tf.float64)
    )


def test_inf_ignored_longer_input(mock_mlm_inf, sampler):
    # One tokenised sequence of length 2, shape: [1, 2]
    sequence = tf.constant([[0, 1, 1]])
    attention_mask = tf.ones_like(sequence)

    input_sequence = {"input_ids": sequence, "attention_mask": attention_mask}

    variants, mut_prob, logits = sampler(
        sequence=input_sequence,
        max_steps=1,
        mask_token=2,
        masked_marginals=False,
        model=mock_mlm_inf,
    )

    assert variants.shape == (1, 3)

    outcome_1 = tf.reduce_all(variants == tf.constant([[0, 1, 0]], dtype=tf.int32))
    outcome_2 = tf.reduce_all(variants == tf.constant([[0, 0, 1]], dtype=tf.int32))

    assert tf.math.logical_or(outcome_1, outcome_2)

    print(mut_prob["step_0"])

    # Input tokens are automatically masked, this means 1. value for position 0 becomes 0
    assert tf.reduce_all(
        mut_prob["step_0"]
        == tf.constant([[[0.0, 0.0], [0.5, 0.0], [0.5, 0.0]]], dtype=tf.float64)
    )


def test_zero_p_probabilities(mock_mlm_zero, sampler):

    sequence = tf.constant([[0, 1, 1]])
    attention_mask = tf.ones_like(sequence)

    input_sequence = {"input_ids": sequence, "attention_mask": attention_mask}
    with pytest.raises(ValueError, match="\nZeros detected in output probabilities."):
        variants, mut_prob, logits = sampler(
            sequence=input_sequence,
            max_steps=1,
            mask_token=2,
            masked_marginals=False,
            model=mock_mlm_zero,
        )
