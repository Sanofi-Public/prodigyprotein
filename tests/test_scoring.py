import pytest
import tensorflow as tf
from prodigyprotein.scoring import (
    masked_marginal_variant_score,
    masked_marginal_independent_variant_score,
    perplexity_score,
    perplexity_ratio_variant_score,
    wildtype_marginal_variant_score,
)


# Test on mock model which produces logits for 2 possible tokens at every pos
class MockMLM:
    def predict(self, inputs):
        # Mock prediction method to return fixed logits for testing
        batch_size, seq_len = tf.shape(inputs["input_ids"])
        zero_logit = tf.zeros(shape=(batch_size, seq_len, 1), dtype=tf.float32)
        ones_logit = tf.ones(shape=(batch_size, seq_len, 1), dtype=tf.float32)
        twos_logit = tf.fill(value=2.0, dims=(batch_size, seq_len, 2))
        logits = tf.concat([zero_logit, ones_logit, twos_logit], axis=-1)
        return {"logits": logits}


class MockMLMMasking:
    def predict(self, inputs):
        # Mock prediction method to return fixed logits for testing
        mask = 4
        batch_size, seq_len = tf.shape(inputs["input_ids"])
        zero_logit = tf.zeros(shape=(batch_size, seq_len, 1), dtype=tf.float32)
        ones_logit = tf.ones(shape=(batch_size, seq_len, 1), dtype=tf.float32)
        twos_logit = tf.fill(value=2.0, dims=(batch_size, seq_len, 2))
        logits = tf.concat([zero_logit, ones_logit, twos_logit], axis=-1)

        # Check masking is working by returning different logits for masked positions
        indices = tf.where(inputs["input_ids"] == mask)
        updates = tf.constant([[1.0, 3.0, 6.0, 9.0]])
        updates = tf.repeat(updates, tf.shape(indices)[0], axis=0)
        logits = tf.tensor_scatter_nd_update(logits, indices=indices, updates=updates)

        return {"logits": logits}


@pytest.fixture
def mock_mlm_masking():
    return MockMLMMasking()


@pytest.fixture
def mock_mlm():
    return MockMLM()


def test_masked_marginal_variant_score_single_wt_multiple_variants(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0], [0, 2, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )
    assert variant_score.shape == (2,)


def test_masked_marginal_variant_score_multiple_wt_multiple_variants(mock_mlm):
    wt_tokens = tf.constant([[2, 0, 1], [3, 0, 2]])
    variant_tokens = tf.constant([[1, 0, 1], [2, 0, 2]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )
    assert variant_score.shape == (2,)


def test_masked_marginal_variant_score_no_substitutions(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    with pytest.raises(ValueError, match="Some variants have no substitutions"):
        masked_marginal_variant_score(wt_tokens, variant_tokens, mask_token, mock_mlm)


def test_masked_marginal_variant_score_different_shapes(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0, 0], [0, 2, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )
    assert variant_score.shape == (2,)


def test_masked_marginal_variant_score_single_mutation_computation(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )

    assert variant_score.shape == (4,)
    assert tf.experimental.numpy.allclose(0.9999999, variant_score)


def test_masked_marginal_variant_score_multiple_mutation_computation(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )

    assert variant_score.shape == (4,)
    assert tf.experimental.numpy.allclose(
        tf.constant(
            [2.0000002, 1.0, 1.0000001, 1.0000001], shape=(4,), dtype=tf.float32
        ),
        variant_score,
    )


def test_masked_marginal_independent_score_no_substitutions(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }
    mask_token = 99

    variant_score = masked_marginal_independent_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )

    assert tf.reduce_all(variant_score == 0)


def test_masked_marginal_independent_score_different_shapes(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0, 0], [0, 2, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_independent_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )

    assert variant_score.shape == (2,)


def test_masked_marginal_independent_single_mutation_computation(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 99
    variant_score = masked_marginal_independent_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm
    )

    assert variant_score.shape == (4,)
    assert tf.experimental.numpy.allclose(0.9999999, variant_score)


def test_masking_masked_marginals_independent_score_multiple_variants(
    mock_mlm_masking,
):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 1], [1, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    mask_token = 4
    variant_score = masked_marginal_independent_variant_score(
        wt_tokens, variant_tokens, mask_token, mock_mlm_masking
    )

    assert variant_score.shape == (2,)
    assert tf.experimental.numpy.allclose(
        tf.constant([4.000001, 2.0000005], shape=(2,), dtype=tf.float32),
        variant_score,
    )


def test_perplexity_score_invalid_shape(mock_mlm):
    tokens = tf.constant([0, 1, 2, 3])

    tokens = {"input_ids": tokens, "attention_mask": tf.ones_like(tokens)}

    with pytest.raises(
        ValueError,
        match="Tokens and attention mask must have shape = \\[n sequences, sequence length\\]",
    ):
        perplexity_score(tokens, mock_mlm)


def test_perplexity_score_single_sequence(mock_mlm):
    tokens = tf.constant([[0, 1, 2, 3]])

    tokens = {"input_ids": tokens, "attention_mask": tf.ones_like(tokens)}

    perplexity = perplexity_score(tokens, mock_mlm)
    assert perplexity.shape == (1,)


def test_perplexity_score_multiple_sequences(mock_mlm):
    tokens = tf.constant([[0, 1, 2, 3], [3, 2, 1, 0]])

    tokens = {"input_ids": tokens, "attention_mask": tf.ones_like(tokens)}

    perplexity = perplexity_score(tokens, mock_mlm)
    assert perplexity.shape == (2,)


def test_perplexity_score_computation(mock_mlm):
    tokens = tf.constant([[2, 2, 2, 2]])

    tokens = {"input_ids": tokens, "attention_mask": tf.ones_like(tokens)}

    perplexity = perplexity_score(tokens, mock_mlm)
    expected_perplexity = tf.constant([2.5032148])
    assert tf.experimental.numpy.allclose(perplexity, expected_perplexity)


def perplexity_ratio_variant_score_single_sequence(mock_mlm):
    wt_tokens = tf.constant([[0, 1, 2, 3]])
    variant_tokens = tf.constant([[0, 1, 2, 3]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    perplexity = perplexity_ratio_variant_score(wt_tokens, variant_tokens, mock_mlm)
    assert perplexity.shape == (1,)


def test_perplexity_ratio_score_multiple_sequences(mock_mlm):
    wt_tokens = tf.constant([[0, 1, 2, 3]])
    variant_tokens = tf.constant([[0, 1, 2, 3], [0, 1, 2, 3]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    perplexity = perplexity_ratio_variant_score(wt_tokens, variant_tokens, mock_mlm)
    assert perplexity.shape == (2,)


def test_perplexity_ratio_score_computation(mock_mlm):
    wt_tokens = tf.constant([[2, 2, 2, 2]])
    variant_tokens = tf.constant([[2, 2, 2, 2]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    perplexity = perplexity_ratio_variant_score(wt_tokens, variant_tokens, mock_mlm)
    expected_perplexity = tf.constant([1])
    assert tf.experimental.numpy.allclose(perplexity, expected_perplexity)


def test_perplexity_ratio_score_computation_2(mock_mlm):
    wt_tokens = tf.constant([[2, 2, 2, 2]])
    variant_tokens = tf.ones_like(wt_tokens)

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    perplexity = perplexity_ratio_variant_score(wt_tokens, variant_tokens, mock_mlm)
    expected_perplexity = tf.constant([2.7182817])
    assert tf.experimental.numpy.allclose(perplexity, expected_perplexity)


def test_wildtype_marginal_variant_score_single_wt_multiple_variants(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0], [0, 2, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    variant_score = wildtype_marginal_variant_score(wt_tokens, variant_tokens, mock_mlm)
    assert variant_score.shape == (2,)


def test_wildtype_marginal_variant_score_multiple_wt_multiple_variants(mock_mlm):
    wt_tokens = tf.constant([[2, 0, 1], [3, 0, 2]])
    variant_tokens = tf.constant([[1, 0, 1], [2, 0, 2]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    variant_score = wildtype_marginal_variant_score(wt_tokens, variant_tokens, mock_mlm)
    assert variant_score.shape == (2,)


def test_wildtype_marginal_variant_score_different_shapes(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0, 0], [0, 2, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    variant_score = wildtype_marginal_variant_score(wt_tokens, variant_tokens, mock_mlm)
    assert variant_score.shape == (2,)


def test_wildtype_marginal_variant_score_single_mutation_computation(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    variant_score = wildtype_marginal_variant_score(wt_tokens, variant_tokens, mock_mlm)

    assert variant_score.shape == (4,)
    assert tf.experimental.numpy.allclose(0.9999999, variant_score)


def test_wildtype_marginal_variant_score_multiple_mutation_computation(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    variant_score = wildtype_marginal_variant_score(wt_tokens, variant_tokens, mock_mlm)

    assert variant_score.shape == (4,)
    assert tf.experimental.numpy.allclose(
        tf.constant(
            [2.0000002, 1.0, 1.0000001, 1.0000001], shape=(4,), dtype=tf.float32
        ),
        variant_score,
    )


def test_wildtype_marginal_variant_score_no_mutation_computation(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    variant_score = wildtype_marginal_variant_score(wt_tokens, variant_tokens, mock_mlm)

    assert variant_score.shape == (4,)
    assert tf.experimental.numpy.allclose(
        0.0,
        variant_score,
    )


def test_wildtype_marginal_variant_score_wrong_tokeniser(mock_mlm):
    wt_tokens = tf.constant([[0, 0, 0]])
    variant_tokens = tf.constant([[0, 0, 10], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    wt_tokens = {"input_ids": wt_tokens, "attention_mask": tf.ones_like(wt_tokens)}
    variant_tokens = {
        "input_ids": variant_tokens,
        "attention_mask": tf.ones_like(variant_tokens),
    }

    with pytest.raises(
        ValueError,
        match="Input tokens contain integers greater than vocab size of model",
    ):
        variant_score = wildtype_marginal_variant_score(
            wt_tokens, variant_tokens, mock_mlm
        )
