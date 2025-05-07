import tensorflow as tf


def wildtype_marginal_variant_score(wt_tokens, variant_tokens, mlm):
    """
    Computes the wildtype marginal score between wildtype sequences
    and variant sequences.

    The wildtype marginal score is the sum of the log ratio of the
    probabilities of the variant tokens and wildtype tokens at each mutated position.

    Parameters
    ----------
    wt_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'.
        Values are tf.Tensors with shape = `[1, sequence length]` or `[n_variants, sequence length]`
    variant_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'.
        Values are tf.Tensors with shape = `[n variants, sequence length]`
    mlm :
        Model that takes tokens as a tf.Tensor of shape `[n_variants, sequence_length]` as input
        and returns an object with logits as a key. Format of models returned by
        Hugging Face TFAutoModelForMaskedLM class.

    Returns
    -------
    variant_score : tf.Tensor
        Wildtype variant score per variant with shape = `[batch_size, ]`.
    """

    wt_tokens, wt_attention_mask = tf.convert_to_tensor(
        wt_tokens["input_ids"], dtype=tf.int32
    ), tf.convert_to_tensor(wt_tokens["attention_mask"], dtype=tf.int32)

    variant_tokens = tf.convert_to_tensor(
        variant_tokens["input_ids"], dtype=wt_tokens.dtype
    )

    # Ensure shape of input tensors is correct
    if tf.rank(wt_tokens) != 2 or tf.rank(variant_tokens) != 2:
        raise ValueError(
            "Input wt and variant tokens must be rank 2 with shape `[batch size, sequence length]`"
        )

    # Inspect dim of input for wt and variant sequences
    batch, seq_len = tf.shape(variant_tokens)
    wt_batch, wt_seq_len = tf.shape(wt_tokens)

    if batch != wt_batch and wt_batch != 1:
        raise ValueError(
            "Values for wt_token must be`[1, sequence length]` or `[n variants, sequence length]`"
        )

    if seq_len != wt_seq_len:
        raise ValueError(
            "Seq length for `wt_tokens` must match `variant_token` seq length"
        )

    # For each variant, identify positions which have changed compared to the wt.
    substitution_mask = tf.not_equal(wt_tokens, variant_tokens)

    # Identify indices where changes have occured compared to the wt.
    position_of_substitutions = tf.where(substitution_mask)
    position_of_substitutions = tf.cast(position_of_substitutions, tf.int32)

    if tf.shape(wt_tokens)[0] == batch:

        position_of_substitution_wt = tf.identity(position_of_substitutions)

    else:
        # The positions will be identical for the wt sequence. Since only one sequence
        # replace the batch indice with 0
        position_of_substitution_wt = tf.concat(
            [
                tf.zeros((position_of_substitutions.shape[0], 1), dtype=tf.int32),
                position_of_substitutions[:, 1:],
            ],
            axis=-1,
        )

    # Construct indices pointing to [variant sequence, mutated position, mutant token]
    # Construct indices pointing to [variant sequence, mutated position, wildtype token]
    tokens_mt = tf.gather_nd(variant_tokens, position_of_substitutions)
    tokens_wt = tf.gather_nd(wt_tokens, position_of_substitution_wt)

    tokens_mt = tf.expand_dims(tokens_mt, axis=-1)
    tokens_wt = tf.expand_dims(tokens_wt, axis=-1)

    pos_ind_token_wt = tf.concat([position_of_substitutions, tokens_wt], axis=-1)
    pos_ind_token_mt = tf.concat([position_of_substitutions, tokens_mt], axis=-1)

    # Construct input for model inference
    input_sequences = {
        "input_ids": wt_tokens,
        "attention_mask": wt_attention_mask,
    }

    # Obtain logits over vocab per residue from the mlm
    logits = mlm.predict(input_sequences)["logits"]

    # Detect vocab size from logit matrix, ensure users passed correct
    # tokens.
    vocab_size = tf.shape(logits)[-1]

    tokens_in_input = tf.concat([tokens_mt, tokens_wt], axis=0)

    tokens_in_input, _ = tf.unique(tf.squeeze(tokens_in_input))

    if tf.reduce_max(tokens_in_input) > vocab_size - 1:
        raise ValueError(
            "Input tokens contain integers greater than vocab size of model"
        )

    # # Convert logits to probabilities
    probs = tf.nn.softmax(logits)

    if not tf.shape(probs)[0] == batch:
        probs = tf.repeat(probs, repeats=batch, axis=0)

    # Initialise tensor to hold probabilities from masked residues (wt and mt token)
    # Use ones. After log these become zero.
    probs_wt_mt = tf.ones((batch, seq_len, 2))

    # # Grab wt and mt token prob for every variant at each mutated position
    mt_prob_update = tf.gather_nd(params=probs, indices=pos_ind_token_mt)
    wt_prob_update = tf.gather_nd(params=probs, indices=pos_ind_token_wt)

    updates = tf.concat([mt_prob_update[:, None], wt_prob_update[:, None]], axis=-1)

    # Update mutated positions with mt and wt prob values
    probs_wt_mt = tf.tensor_scatter_nd_update(
        probs_wt_mt, position_of_substitutions, updates
    )

    # Obtain log probabilities, all unmutated positions become zero as well.
    log_prob_wt_mt = tf.math.log(probs_wt_mt)

    # Calculate the log ratio between for each mutated residue (for each variant)
    wt_marginals = log_prob_wt_mt[:, :, 0] - log_prob_wt_mt[:, :, 1]

    # Sum the masked log ratios between mt and wt for each variant.
    variant_score = tf.reduce_sum(wt_marginals, axis=-1)

    return variant_score


def masked_marginal_variant_score(wt_tokens, variant_tokens, mask_token, mlm):
    """
    Computes the masked marginal score between wildtype sequences
    and variant sequences.

    All mutated positions are masked at the same time for calculation of the masked marginal score,
    for a single variant sequence. The masked marginal score is the sum of the log ratio of the
    probabilities of the variant tokens and wildtype tokens at each mutated position.

    Examples
    --------
    **All variants of one wildtype sequence**
    >>> variant_tokens = {'input_ids': tf.constant([[0, 1, 0]]), 'attention_mask': tf.constant(tf.constant([[1, 1, 1]]))}
    >>> wt_tokens = {'input_ids': tf.constant([[0, 0, 0]]), 'attention_mask': tf.constant(tf.constant([[1, 1, 1]]))}
    >>> mask_token = 32
    >>> variant_score = directed_evolution.masked_marginal_variant_score(wt_tokens, variant_tokens, mask_token, mlm)

    **Each variant has its own corresponding wildtype sequence**
    >>> variant_tokens = tf.constant([[1, 0, 1], [2, 0, 2]])
    >>> wt_tokens = {'input_ids': tf.constant([[0, 1, 0], [1, 1, 0]]), 'attention_mask': tf.constant(tf.constant([[1, 1, 1], [1, 1, 1]]))}
    >>> mask_token = 32
    >>> variant_score = directed_evolution.masked_marginal_variant_score(wt_tokens, variant_tokens, mask_token, mlm)

    Parameters
    ----------
    wt_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'. Values are tf.Tensors with shape = `[1, sequence length]` or `[n variants, sequence length]`
    variant_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'. Values are tf.Tensors with shape = `[n variants, sequence length]`
    mask_token : int
        The integer representing the special mask token in the masked language model vocab.
    mlm :
        Model that can take a dict with keys `input_ids` and `attention_mask`. Both `input_ids` and `attention_mask` as a tf.Tensor of shape
        `[n_variants, sequence_length]`. Returns a object where logits can be accessed with the key `logits`. Format of models returned by
        Hugging Face TFAutoModelForMaskedLM class.

    Returns
    -------
    variant_score : tf.Tensor
        Masked marginal variant score per variant with shape = `[batch_size, ]`.
    """
    wt_tokens, wt_attention_mask = tf.convert_to_tensor(
        wt_tokens["input_ids"], dtype=tf.int32
    ), tf.convert_to_tensor(wt_tokens["attention_mask"], dtype=tf.int32)
    variant_tokens = tf.convert_to_tensor(
        variant_tokens["input_ids"], dtype=wt_tokens.dtype
    )

    # Inspect dim of input for wt and variant sequences
    batch, seq_len = tf.shape(variant_tokens)
    wt_batch, wt_seq_len = tf.shape(wt_tokens)

    if batch != wt_batch and wt_batch != 1:
        raise ValueError(
            "Values for wt_token must be`[1, sequence length]` or `[n variants, sequence length]`"
        )

    # For each variant, identify positions which have changed compared to the wt.
    substitution_mask = tf.not_equal(wt_tokens, variant_tokens)

    # Identify indices where changes have occured compared to the wt.
    position_of_substitutions = tf.where(substitution_mask)
    position_of_substitutions = tf.cast(position_of_substitutions, tf.int32)

    if tf.shape(wt_tokens)[0] == batch:

        position_of_substitution_wt = tf.identity(position_of_substitutions)

    else:
        # The positions will be identical for the wt sequence. Since only one sequence
        # replace the batch indice with 0
        position_of_substitution_wt = tf.concat(
            [
                tf.zeros((position_of_substitutions.shape[0], 1), dtype=tf.int32),
                position_of_substitutions[:, 1:],
            ],
            axis=-1,
        )

        # Since many variants to one wt sequence, need to copy wt to match number
        # of variant sequences
        wt_attention_mask = tf.repeat(wt_attention_mask, repeats=batch, axis=0)

    # Construct indices pointing to [variant sequence, mutated position, mutant token]
    # Construct indices pointing to [variant sequence, mutated position, wildtype token]
    tokens_mt = tf.gather_nd(variant_tokens, position_of_substitutions)
    tokens_wt = tf.gather_nd(wt_tokens, position_of_substitution_wt)

    tokens_mt = tf.expand_dims(tokens_mt, axis=-1)
    tokens_wt = tf.expand_dims(tokens_wt, axis=-1)

    pos_ind_token_wt = tf.concat([position_of_substitutions, tokens_wt], axis=-1)
    pos_ind_token_mt = tf.concat([position_of_substitutions, tokens_mt], axis=-1)

    # For each variant, construct a masked wildtype tokenised input sequence
    mask_tensor = tf.fill(tf.shape(variant_tokens), mask_token)

    masked_input_sequences = tf.where(
        variant_tokens == wt_tokens, variant_tokens, mask_tensor
    )

    substitutions_per_variant = tf.reduce_sum(
        tf.cast(masked_input_sequences == mask_token, tf.int32), axis=-1
    )
    no_substitutions = tf.reduce_any(tf.equal(substitutions_per_variant, 0))

    if no_substitutions:
        raise ValueError(
            "Pass only variants to the `variant_tokens` argument. Some variants have no substitutions."
        )

    # Construct input for model inference
    masked_input_sequences = {
        "input_ids": masked_input_sequences,
        "attention_mask": wt_attention_mask,
    }

    # Obtain logits over vocab per residue from the mlm
    logits = mlm.predict(masked_input_sequences)["logits"]

    # Convert logits to probabilities
    probs = tf.nn.softmax(logits)

    # Initialise tensor to hold probabilities from masked residues (wt and mt token)
    # Use ones. After log these become zero.
    probs_wt_mt = tf.ones((batch, seq_len, 2))

    # Grab wt and mt token prob for every variant at each mutated position
    mt_prob_update = tf.gather_nd(params=probs, indices=pos_ind_token_mt)
    wt_prob_update = tf.gather_nd(params=probs, indices=pos_ind_token_wt)
    updates = tf.concat([mt_prob_update[:, None], wt_prob_update[:, None]], axis=-1)

    # Update mutated positions with mt and wt prob values
    probs_wt_mt = tf.tensor_scatter_nd_update(
        probs_wt_mt, position_of_substitutions, updates
    )

    # Obtain log probabilities, all unmutated positions become zero as well.
    log_prob_wt_mt = tf.math.log(probs_wt_mt)

    # Calculate the log ratio between for each mutated residue (for each variant)
    masked_marginals = log_prob_wt_mt[:, :, 0] - log_prob_wt_mt[:, :, 1]

    # Sum the masked log ratios between mt and wt for each variant.
    variant_score = tf.reduce_sum(masked_marginals, axis=-1)

    return variant_score


def masked_marginal_independent_variant_score(
    wt_tokens, variant_tokens, mask_token, mlm
):
    """
    Computes the masked marginal score (one substitution at a time) between wildtype sequences
    and variant sequences.

    Mutated positions are masked independently for calculation of the masked marginal score,
    for a single variant sequence. The masked marginal score is the sum of the log ratio of the
    probabilities of the variant tokens and wildtype tokens at each mutated position.

    Parameters
    ----------
    wt_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'.
        Values are tf.Tensors with shape = `[1, sequence length]`
    variant_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'.
        Values are tf.Tensors with shape = `[n variants, sequence length]`
    mask_token : int
        The integer representing the special mask token in the masked language models vocab.
    mlm :
        Model that takes tokens as a tf.Tensor of shape `[n_variants, sequence_length]` as input
        and returns an object with logits as a key. Format of models returned by
        Hugging Face TFAutoModelForMaskedLM class.

    Returns
    -------
    variant_score : tf.Tensor
        Masked marginal variant score per variant with shape = `[batch_size, ]`.
    """
    wt_tokens, wt_attention_mask = tf.convert_to_tensor(
        wt_tokens["input_ids"], dtype=tf.int32
    ), tf.convert_to_tensor(wt_tokens["attention_mask"], dtype=tf.int32)
    variant_tokens = tf.convert_to_tensor(
        variant_tokens["input_ids"], dtype=wt_tokens.dtype
    )

    # Ensure shape of input tensors is correct
    if tf.rank(wt_tokens) != 2 or tf.rank(variant_tokens) != 2:
        raise ValueError(
            "Input wt and variant tokens must be rank 2 with shape `[batch size, sequence length]`"
        )

    if tf.shape(wt_tokens)[0] != 1:
        raise ValueError(
            "Wild type tokenised sequence values must be shape = `[1, sequence length]`"
        )

    # Inspect dim of input for wt and variant sequences
    batch, seq_len = tf.shape(variant_tokens)
    wt_batch, wt_seq_len = tf.shape(wt_tokens)

    if seq_len != wt_seq_len:
        raise ValueError(
            "Wildtype and variant sequences must have the same sequence length"
        )

    # Identify unique positions mutated across all variants
    mutated_positions = tf.where(wt_tokens != variant_tokens)[:, 1]
    unique_positions, duplicate_map = tf.unique(mutated_positions)

    # Identify how many times model inference needs to be run (one per unique position)
    number_of_repeats = tf.shape(unique_positions)
    # Create input for model inference where batch size equals number of unique mutated positions
    masked_wt = tf.repeat(wt_tokens, number_of_repeats, axis=0)

    # Create indices and updates that will be used to update input tensor with masked token
    # There will only be one mask token in each sequence (unless user passed input with mask token)
    indices = tf.concat(
        [
            tf.range(number_of_repeats)[:, None],
            tf.cast(unique_positions[:, None], tf.int32),
        ],
        axis=1,
    )
    updates = tf.repeat(tf.constant([mask_token]), number_of_repeats)

    # Update input tensor with mask token at each unique mutated position
    masked_wt = tf.repeat(wt_tokens, number_of_repeats, axis=0)
    masked_wt = tf.tensor_scatter_nd_update(masked_wt, indices=indices, updates=updates)

    wt_attention_mask = tf.repeat(wt_attention_mask, repeats=number_of_repeats, axis=0)

    # Construct model input
    model_input = {"input_ids": masked_wt, "attention_mask": wt_attention_mask}

    # Obtain logits over vocab per residue from the mlm
    logits_for_masked_positions = mlm.predict(model_input)["logits"]

    # Convert logits to probabilities
    prob_for_masked_position = tf.nn.softmax(logits_for_masked_positions)

    # Initialise tensor to hold probabilities from masked residues (wt and mt token)
    mutated_positions = tf.where(wt_tokens != variant_tokens)
    matrix = tf.ones(
        shape=(tf.shape(variant_tokens)[0], tf.shape(variant_tokens)[1], 2)
    )

    # Grab wt and mt tokens for every variant at each mutated position
    mutated_token = tf.gather_nd(params=variant_tokens, indices=mutated_positions)[
        :, None
    ]
    wt_copy = tf.repeat(wt_tokens, tf.shape(variant_tokens)[0], axis=0)
    wt_token = tf.gather_nd(params=wt_copy, indices=mutated_positions)[:, None]

    wt_residue_indices = tf.concat(
        [
            tf.cast(duplicate_map, tf.int32)[:, None],
            tf.cast(mutated_positions[:, 1][:, None], tf.int32),
        ],
        axis=1,
    )
    wt_residue_indices = tf.concat(
        [tf.cast(wt_residue_indices, tf.int32), wt_token], axis=1
    )

    substitute_residue_indices = tf.concat(
        [
            tf.cast(duplicate_map, tf.int32)[:, None],
            tf.cast(mutated_positions[:, 1][:, None], tf.int32),
        ],
        axis=1,
    )
    substitute_residue_indices = tf.concat(
        [substitute_residue_indices, mutated_token], axis=1
    )

    # Grab wt and mt token prob for every variant at each mutated position
    wt_masked_logits = tf.gather_nd(
        params=prob_for_masked_position, indices=wt_residue_indices
    )[:, None]
    substitute_masked_logits = tf.gather_nd(
        params=prob_for_masked_position, indices=substitute_residue_indices
    )[:, None]
    updates_for_each_variant = tf.concat(
        [substitute_masked_logits, wt_masked_logits], axis=1
    )

    # Update mutated positions with mt and wt prob values
    masked_prob_per_sequence = tf.tensor_scatter_nd_update(
        matrix, indices=mutated_positions, updates=updates_for_each_variant
    )

    # Obtain log probabilities, all unmutated positions become zero as well.
    masked_prob_per_sequence = tf.math.log(masked_prob_per_sequence)

    # Calculate the log ratio between each mutated residue and the wt residue (for each variant)
    llr = tf.subtract(
        masked_prob_per_sequence[:, :, 0], masked_prob_per_sequence[:, :, 1]
    )

    # Sum the masked log ratios between mt and wt for each variant.
    llr_per_sequence = tf.reduce_sum(llr, axis=-1)

    return llr_per_sequence


def perplexity_score(tokens, mlm):
    """
    Computes the perplexity score of a tokenised sequence for a given masked language model.

    Examples
    --------
    **Many tokenised sequences**
    >>> perplexity = perplexity_score(wt_tokens, mlm)

    Parameters
    ----------
    tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'.
        Values are tf.Tensors with shape = `[n variants, sequence length]`
    mlm :
        Model that takes dictionary with keys 'input_ids' and 'attention_mask' whose values
        are tf.Tensors of shape `[n_variants, sequence_length]`.
        Must return an object with `logits` as a key. Format of models returned by
        Hugging Face TFAutoModelForMaskedLM class.

    Returns
    -------
    perplexity_score : tf.Tensor
        Perplexity score per tokenised sequence with shape = `[batch_size, ]`.
    """
    tokens, attention_mask = tokens["input_ids"], tokens["attention_mask"]

    # Ensure values input are int32 (and tf.Tensors)
    tokens = tf.cast(tokens, tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)

    # Check rank of input
    if tf.rank(tokens) != 2 or tf.rank(attention_mask) != 2:
        raise ValueError(
            "Tokens and attention mask must have shape = [n sequences, sequence length]"
        )

    sce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    # Construct model input
    model_input = {"input_ids": tokens, "attention_mask": attention_mask}

    # Calculate perplexity
    sce = sce_loss(tokens, mlm.predict(model_input)["logits"])

    sce = tf.reduce_mean(sce, axis=-1)

    perplexity_score = tf.exp(sce)

    return perplexity_score


def perplexity_ratio_variant_score(wt_tokens, variant_tokens, mlm):
    """
    Computes the ratio for the perplexity scores between a tokenised wildtype sequence
    and tokenised variant sequences for a given masked language model.

    Examples
    --------
    **All variants of one wildtype sequence**
    >>> variant_tokens = {'input_ids': tf.constant([[0, 1, 0]]), 'attention_mask': tf.constant(tf.constant([[1, 1, 1]]))}
    >>> wt_tokens = {'input_ids': tf.constant([[0, 0, 0]]), 'attention_mask': tf.constant(tf.constant([[1, 1, 1]]))}
    >>> variant_score = perplexity_ratio_variant_score(wt_tokens, variant_tokens, mlm)

    **Each variant has its own corresponding wildtype sequence**
    >>> variant_tokens = tf.constant([[1, 0, 1], [2, 0, 2]])
    >>> wt_tokens = {'input_ids': tf.constant([[0, 1, 0], [1, 1, 0]]), 'attention_mask': tf.constant(tf.constant([[1, 1, 1], [1, 1, 1]]))}
    >>> variant_score = perplexity_ratio_variant_score(wt_tokens, variant_tokens, mlm)

    Parameters
    ----------
    wt_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'. Values are tf.Tensors with shape = `[1, sequence length]` or `[n variants, sequence length]`
    variant_tokens : dict
        Dictionary with keys 'input_ids' and 'attention_mask'. Values are tf.Tensors with shape = `[n variants, sequence length]`
    mlm :
        Model that takes tokens as a tf.Tensor of shape `[n_variants, sequence_length]` as input
        and returns a dictionary with logits as a key. Format of models returned by
        Hugging Face TFAutoModelForMaskedLM class.

    Returns
    -------
    perplexity_ratio_variant_score : tf.Tensor
        Ratio of variant perplexity score against wildtype sequence with shape = `[batch_size, ]`.
    """
    wt_tokens, wt_attention_mask = tf.convert_to_tensor(
        wt_tokens["input_ids"]
    ), tf.convert_to_tensor(wt_tokens["attention_mask"])
    variant_tokens, variant_attention_mask = tf.convert_to_tensor(
        variant_tokens["input_ids"]
    ), tf.convert_to_tensor(variant_tokens["attention_mask"])

    if tf.rank(wt_tokens) != 2 or tf.rank(variant_tokens) != 2:
        raise ValueError(
            "input ids and attention masks must be dim `[n variants, sequence length]` "
        )

    wt_batch, _ = tf.shape(wt_tokens)
    mt_batch, _ = tf.shape(variant_tokens)

    if wt_batch != 1 and wt_batch != mt_batch:
        raise ValueError(
            "Either one wildtype sequence for comparison with all variants, or one wildtype sequence for every variant"
        )

    # Construct model input
    wt_model_input = {"input_ids": wt_tokens, "attention_mask": wt_attention_mask}

    variant_model_input = {
        "input_ids": variant_tokens,
        "attention_mask": variant_attention_mask,
    }

    wt_perplexity = perplexity_score(wt_model_input, mlm)
    variant_perplexity = perplexity_score(variant_model_input, mlm)

    perplexity_ratio_variant_score = variant_perplexity / wt_perplexity

    return perplexity_ratio_variant_score
