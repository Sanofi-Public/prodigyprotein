import tensorflow as tf
from .blosum_embedding import BlosumProbabilityEmbedding
from warnings import warn
import logging


class DirectedEvolutionSampler:
    """Base sampler class.

    Args:
        temperature: float. optional. Used to control the
            randomness of the sampling. The higher the temperature, the
            more diverse the samples. Defaults to `1.0`.

    Call arguments:
        {{call_args}}

    This base class can be extended to implement different directed evolution
    sampling methods. To do so, override the `get_next_mutation()` method, which
    computes the next token to mutate based on a probability distribution over all
    possible vocab entries.

    """

    def __init__(self, temperature=1.0, llr_threshold=None):

        self.temperature = (
            temperature if isinstance(temperature, float) else float(temperature)
        )

        if llr_threshold is not None:
            llr_threshold = (
                llr_threshold
                if isinstance(llr_threshold, float)
                else float(llr_threshold)
            )

        self.llr_threshold = llr_threshold

    def __call__(
        self,
        sequence,
        model,
        max_steps: int,
        mask_token: int,
        masked_marginals=True,
        start_end_intervals=None,
        batch=32,
        # blosum=None,
        # reorder_blosum=None
    ):

        batch = batch if isinstance(batch, int) else int(batch)
        protein_sequence = sequence["input_ids"]
        attention_mask = sequence["attention_mask"]

        pbar = tf.keras.utils.Progbar(
            target=max_steps, stateful_metrics=[], unit_name="Directed Evolution Steps"
        )

        max_steps = tf.cast(max_steps, "int32")
        mask_token = tf.cast(mask_token, protein_sequence.dtype)

        # Initialise plausible mutations
        plausible_mutations = True

        # Validate start end intervals if provided
        if start_end_intervals is not None:
            start_end_intervals = self._validate_intervals(start_end_intervals)

        # Default method for calculating logits is masked marginals
        if masked_marginals:
            logging.info("Logits computed using masked tokens")

            def _next(protein_sequence, attention_mask):
                """
                Computes masked marginals for all positions in one tokenised sequence.
                Only runs unique sequences in batch through. Logits (results) are then mapped back to the duplicate entries.

                This function repeats the input tensor sequence length times or interval length sum times,
                generates a model input template, sets diagonal entries to mask token to generate masked logit entries
                and computes logits. It also retrieves logits for masked token at each position and reshapes the logits.

                Parameters
                ----------
                protein_sequence : tf.Tensor
                    The protein sequence to be processed.
                attention_mask : tf.Tensor
                    The attention mask for the protein sequence.

                Returns
                -------
                tf.Tensor
                    The reshaped masked marginal logits for the protein sequence.

                Notes
                -----
                This function performs batch model inference to prevent Out Of Memory (OOM) errors.
                The batch size and sequence length are determined from the shape of the protein sequence.
                The logits are computed for each batch and then concatenated.
                """
                # Retrieve only unique protein sequences
                protein_sequence, gather_idx = tf.raw_ops.UniqueV2(
                    x=protein_sequence, axis=[0]
                )

                # Get input seq length
                batch_size, seq_len = tf.shape(protein_sequence)

                # Need to retrieve the attention tensors for model inference
                att_gather_idx = tf.concat(
                    [tf.where(gather_idx == i)[0] for i in tf.range(batch_size)], axis=0
                )

                attention_mask = tf.gather(
                    params=attention_mask, indices=att_gather_idx
                )

                if start_end_intervals is not None:
                    n_repeats = sum(
                        [interval[1] - interval[0] for interval in start_end_intervals]
                    )
                    # Repeat input tensor interval length times
                    repeats = tf.repeat(n_repeats, repeats=batch_size)

                    # indices to mask
                    indices_to_mask = tf.concat(
                        [
                            tf.range(interval[0], interval[1])
                            for interval in start_end_intervals
                        ],
                        axis=0,
                    )
                else:
                    # Repeat input tensor seq len times

                    repeats = tf.repeat(seq_len, repeats=batch_size)
                    indices_to_mask = tf.range(seq_len)

                # Generate model input template
                protein_sequence = tf.repeat(
                    input=protein_sequence, repeats=repeats, axis=0
                )

                # Repeat attention mask to match repeat tensor

                attention_mask = tf.repeat(
                    input=attention_mask, repeats=repeats, axis=0
                )

                # Generate row indices
                row_indices = tf.range(len(protein_sequence))[:, None]

                # Replicate for all sequences in batch
                indices_to_mask = tf.tile(indices_to_mask, [batch_size])[:, None]

                # indices_to_update
                indices_to_update = tf.concat([row_indices, indices_to_mask], axis=1)

                # Add mask token to specified indices
                protein_sequence = tf.tensor_scatter_nd_update(
                    protein_sequence,
                    updates=[mask_token] * len(protein_sequence),
                    indices=indices_to_update,
                )

                # Batch model inference to prevent OOM
                logits = []

                for i in range(0, len(protein_sequence), batch):

                    logits_ = model(
                        {
                            "input_ids": protein_sequence[i : i + batch],
                            "attention_mask": attention_mask[i : i + batch],
                        }
                    )["logits"]

                    mini_batch_size = tf.shape(logits_)[0]

                    # Generate indices to gather (we want diagonal entries from logit matrix where the mask token was set.
                    indices = tf.expand_dims(tf.range(mini_batch_size), axis=1)
                    indices = tf.concat(
                        [indices, indices_to_mask[i : i + mini_batch_size]], axis=1
                    )

                    # Retrieve logits for masked token at each position
                    logits_ = tf.gather_nd(
                        params=logits_, indices=indices, batch_dims=0
                    )

                    logits.append(logits_)

                logits = tf.concat(logits, axis=0)

                logits = tf.reshape(logits, (batch_size, repeats[0], -1))

                # If intervals are used need to pad logits to seq len.
                # INTERVALS MUST BE SORTED.
                if start_end_intervals is not None:

                    end = 0
                    logit_indices_pos = 0
                    logits_padded = []

                    for i in start_end_intervals:
                        interval_length = i[1] - i[0]
                        zero_pad = tf.zeros(
                            shape=(batch_size, i[0] - end, tf.shape(logits_)[-1])
                        )
                        logit_ext = logits[
                            :,
                            logit_indices_pos : logit_indices_pos + interval_length,
                            :,
                        ]
                        logit_indices_pos += interval_length

                        logits_padded.append(zero_pad)
                        logits_padded.append(logit_ext)
                        end = i[1]

                    end_pad = tf.zeros(
                        shape=(batch_size, seq_len - end, tf.shape(logits_)[-1])
                    )

                    logits_padded.append(end_pad)

                    # Generate padded logits.
                    logits = tf.concat(logits_padded, axis=1)

                # # Map logits to back to input sequences
                logits = tf.gather(params=logits, indices=gather_idx)

                return logits

        # Otherwise use wt marginals (no masked tokens)
        else:

            def _next(protein_sequence, attention_mask):
                """
                Computes wild type marginals for all positions in one tokenised sequence. Fast.

                Parameters
                ----------
                protein_sequence : tf.Tensor
                    The protein sequence to be processed.
                attention_mask : tf.Tensor
                    The attention mask for the protein sequence.

                Returns
                -------
                tf.Tensor
                    The logits for the protein sequence.

                Notes
                -----
                This function performs batch model inference to prevent Out Of Memory (OOM) errors.
                The batch size and sequence length are determined from the shape of the protein sequence.
                The logits are computed for each batch and then concatenated.
                """
                # Retrieve only unique protein sequences
                protein_sequence, gather_idx = tf.raw_ops.UniqueV2(
                    x=protein_sequence, axis=[0]
                )

                batch_size, seq_len = tf.shape(protein_sequence)

                # Need to retrieve the attention tensors for model inference
                att_gather_idx = tf.concat(
                    [tf.where(gather_idx == i)[0] for i in tf.range(batch_size)], axis=0
                )

                attention_mask = tf.gather(
                    params=attention_mask, indices=att_gather_idx
                )

                # Batch model inference to prevent OOM
                logits = []

                for i in range(0, batch_size, batch):

                    logits.append(
                        model(
                            {
                                "input_ids": protein_sequence[i : i + batch],
                                "attention_mask": attention_mask[i : i + batch],
                            }
                        )["logits"]
                    )

                logits = tf.concat(logits, axis=0)

                # Map logits to back to input sequences
                logits = tf.gather(params=logits, indices=gather_idx)

                return logits

        # Use mask from input
        # mask = tf.logical_not(attention_mask == padding_token)
        # The attention mask tells us which residues to pay attention to. Ignore if 0.
        mask = attention_mask == 0

        # Initialise probabilities cache,
        # batch, _, emb_dim = tf.shape(model(protein_sequence).logits[:1])
        # seq_len = tf.argmax(mask,
        #                     axis=-1)
        # seq_len = tf.squeeze(seq_len)
        # probabilities = tf.zeros(shape=(batch, seq_len, emb_dim), dtype='float64')

        # Alternative approach, simply overwrite masked positions with probability=0
        # Do not process entire batch to check output dimension
        total_batch_size = tf.shape(protein_sequence)[0]
        example_output = model((protein_sequence[:1], attention_mask[:1]))["logits"]

        # Copy example output to make batch dimension correct
        example_output = tf.repeat(example_output, total_batch_size, axis=0)

        probabilities = tf.zeros_like(example_output, dtype="float64")
        logits = tf.zeros_like(example_output, dtype=example_output.dtype)

        pbar.add(0)

        def cond(
            protein_sequence, attention_mask, probabilities, logits, plausible_mutations
        ):
            """
            Placeholder condition function for a while loop in tensorflow.

            This function always returns True, allowing the loop to continue indefinitely.
            The loop's actual exit condition must be handled by the loop itself.

            Parameters
            ----------
            protein_sequence : tf.Tensor
                The protein sequence to be processed.
            attention_mask : tf.Tensor
                The attention mask for the protein sequence.
            probabilities : tf.Tensor
                The probabilities associated with the protein sequence.
            logits : tf.Tensor
                The logits computed for the protein sequence.

            Returns
            -------
            bool
                Always returns True.

            Notes
            -----
            This function is a placeholder and its parameters are not used.
            """
            if plausible_mutations is True:
                return True
            else:
                return False

        def body(
            protein_sequence, attention_mask, probabilities, logits, plausible_mutations
        ):
            """
            The main body of a loop for conducting directed evolution of protein sequences.

            This function computes the softmax distribution for the next token, calculates
            log likelihoods ratios (LLR) for each residue compared to the residue in the sequence,
            converts LLRs to mutation probabilities across all positions and tokens, samples the next mutation,
            updates the protein sequence with mutation, and concatenates the probabilities and logits.

            Parameters
            ----------
            protein_sequence : tf.Tensor
                The protein sequence to be processed.
            attention_mask : tf.Tensor
                The attention mask for the protein sequence.
            probabilities : tf.Tensor
                The probabilities associated with the protein sequence.
            logits : tf.Tensor
                The logits computed for the protein sequence.

            Returns
            -------
            tuple
                The updated protein sequence, attention mask, probabilities, logits, start, and end.

            Notes
            -----
            This function is part of a loop and is called repeatedly until a certain condition is met.
            The current residue is always masked in the mutational probabilities, meaning that new tokens are chosen.
            """
            # Record dim
            batch_size, seq_len = tf.shape(protein_sequence)

            # Compute the softmax distribution for the next token.
            logits_ = _next(protein_sequence, attention_mask)

            # If model logits are probabilities skip probability calculation

            # Construct mask to detect where inf values are.
            is_inf_mask = tf.math.is_inf(logits_)
            is_inf_mask = tf.cast(is_inf_mask, tf.bool)

            # Replace inf with zero for inferring whether probabilities are returned
            inf_masked_logits = tf.where(is_inf_mask, tf.zeros_like(logits_), logits_)

            # Detect bounds of all logits
            max_logits = tf.reduce_max(inf_masked_logits, axis=-1)
            min_logits = tf.reduce_min(inf_masked_logits, axis=-1)

            max_less_than_1 = tf.reduce_all(max_logits <= 1)
            min_equal_or_greater_than_0 = tf.reduce_all(min_logits >= 0)

            are_prob = tf.math.logical_and(max_less_than_1, min_equal_or_greater_than_0)

            if not are_prob:

                # Compute probabilities of each token, across all positions
                probabilities_ = self._compute_probabilities(logits_)

            else:
                logging.warning(
                    """Detected probabilities output from model. 
                    Is this expected? Skipping softmax op.
                    """
                )
                # Assign output as probabilities (model returned probabilities).
                probabilities_ = tf.identity(logits_)

            # Check no probabilities_ are equal to 0
            absolute_zero_present = tf.equal(probabilities_, 0)

            if tf.reduce_any(absolute_zero_present):
                raise ValueError("\nZeros detected in output probabilities.")

            # Translate to mutation probability across all positions and tokens
            probabilities_, plausible_mutations = self._compute_mutation_probabilities(
                protein_sequence, probabilities_, mask, start_end_intervals
            )

            if not plausible_mutations:
                logging.info("\nNo plausible mutations. Ending directed evolution.")
                probabilities = tf.concat([probabilities, probabilities_], axis=0)
                logits = tf.concat([logits, logits_], axis=0)
                return (
                    protein_sequence,
                    attention_mask,
                    probabilities,
                    logits,
                    plausible_mutations,
                )

            # Compute the next mutation, this method is different depending on the class.
            index, next_mutation = self._get_next_mutation(probabilities_)

            # Possible some sequences have mutational prob all 0. These sequences should not change.
            # When sampling when all p is 0, index token and next mutation become 0, the
            # below function returns identical residues at positon 0 for those sequences
            # allowing batched processing.
            index, next_mutation = self._skip_mutation(
                probabilities=probabilities_,
                protein_sequence=protein_sequence,
                index=index,
                next_mutation=next_mutation,
            )

            # Create mask, ones at index locations for each sequence
            replace_token_mask = tf.one_hot(index, depth=seq_len)
            replace_token_mask = tf.cast(replace_token_mask, "bool")

            # Generate matrix with new tokens for each sequence
            new_tokens = tf.repeat(next_mutation, repeats=[seq_len])
            new_tokens = tf.reshape(new_tokens, (batch_size, seq_len))
            new_tokens = tf.cast(new_tokens, protein_sequence.dtype)

            # Update protein sequence with mutation
            protein_sequence = tf.where(
                replace_token_mask, new_tokens, protein_sequence
            )

            # Model may return probabilities of type float32... ensure
            # match or concat operation will throw error
            probabilities_ = tf.cast(probabilities_, dtype=probabilities.dtype)
            probabilities = tf.concat([probabilities, probabilities_], axis=0)
            logits = tf.concat([logits, logits_], axis=0)

            pbar.add(1)

            # Return the next prompt, cache and incremented index.
            return (
                protein_sequence,
                attention_mask,
                probabilities,
                logits,
                plausible_mutations,
            )

        protein_sequence, attention_mask, probabilities, logits, plausible_mutations = (
            self._run_loop(
                cond,
                body,
                loop_vars=(
                    protein_sequence,
                    attention_mask,
                    probabilities,
                    logits,
                    plausible_mutations,
                ),
                maximum_steps=max_steps,
            )
        )

        # Record batch size, using to assign probabilities per step
        batch_size = tf.shape(protein_sequence)[0]
        # First batch is empty
        probabilities = probabilities[batch_size:]
        logits = logits[batch_size:]

        probabilities = {
            "step_{}".format(step): probabilities[i : i + batch_size]
            for step, i in enumerate(range(0, tf.shape(probabilities)[0], batch_size))
        }
        logits = {
            "step_{}".format(step): logits[i : i + batch_size]
            for step, i in enumerate(range(0, tf.shape(logits)[0], batch_size))
        }

        return protein_sequence, probabilities, logits

    def _compute_probabilities(self, logits):
        """
        Compute token probabilities from logits.

        This method computes the softmax of the logits divided by the temperature.
        The logits are first cast to float64 to ensure full precision during the computation,
        regardless of their original data type.

        Parameters
        ----------
        logits : tf.Tensor
            The logits for which to compute probabilities.

        Returns
        -------
        tf.Tensor
            The computed probabilities for each token.

        Notes
        -----
        The temperature variable used in this method is not defined in the provided code.
        It should be a class attribute.
        """
        logits = tf.cast(logits, "float64")

        return tf.nn.softmax(logits / self.temperature)

    def _compute_mutation_probabilities(
        self, protein_sequence, probabilities, mask, start_end_intervals
    ):
        """
        Compute probability of substitution mutations.

        This method computes the log probabilities of the input probabilities, calculates the log likelihood ratios of
        mutant to wild type residue across all positions, computes the softmax over all log likelihood ratios,
        applies a mask before the start and after the end positions (if defined) and ensures that the final probabilities sum to 1.

        Parameters
        ----------
        protein_sequence : tf.Tensor
            The protein sequence to be processed.
        probabilities : tf.Tensor
            The probabilities associated with the protein sequence.
        mask : tf.Tensor
            The mask to be applied to the protein sequence.
        start_end_intervals: List[Tuple]
            Optional. None if not provided, otherwise list of start and end indices (integers).

        Returns
        -------
        tf.Tensor
            The computed mutation probabilities for each token.

        Raises
        ------
        ValueError
            If start is greater than end, or if start or end is greater than the sequence length.

        Notes
        -----
        This method assumes that the input probabilities are already normalized and that the mask is a boolean tensor
        of the same shape as the protein sequence. Original tokens in the sequence are masked to ensure that new tokens are chosen.
        """
        # initialise value for plausible mutations
        plausible_mutations = True
        # Get indices of current tokens
        indices = [
            [
                [j, i, token.numpy()]
                for j, seq in enumerate(protein_sequence)
                for i, token in enumerate(seq)
            ]
        ]

        # Detect -inf values, model may output
        is_inf_mask = tf.math.is_inf(probabilities)

        # Convert probabilities to log probability
        log_probabilities = tf.math.log(probabilities)

        # After log, inf values --> nan, convert to zero.
        log_probabilities = tf.where(
            is_inf_mask,
            tf.zeros_like(log_probabilities, dtype=log_probabilities.dtype),
            log_probabilities,
        )

        # batch, seq_len, vocab_size
        batch_size, seq_len, vocab_size = tf.shape(probabilities)

        # Grab probability of wt residue across each position in sequence
        wt_log_probabilities = tf.gather_nd(indices=indices, params=log_probabilities)

        # Reshape to add batch, Add last dimension, to enable broadcasting
        wt_log_probabilities = tf.reshape(
            wt_log_probabilities, (batch_size, seq_len, 1)
        )

        # Calculate log likelihood ratios of mt to wt residue across all positions
        mt_llr = tf.math.subtract(log_probabilities, wt_log_probabilities)

        # Convert zeros back to -inf present in the input probabilities.
        mt_llr = tf.where(is_inf_mask, probabilities, mt_llr)

        # If start defined, mask all regions outside intervals
        if start_end_intervals is not None:

            mt_llr = self._mask_intervals(mt_llr, start_end_intervals)

        # If llr threshold set, make all below it 0 after softmax.
        if self.llr_threshold is not None:

            llr_mask = mt_llr >= self.llr_threshold
            inf_matrix = tf.fill(tf.shape(mt_llr), tf.math.log(0.0))
            inf_matrix = tf.cast(inf_matrix, mt_llr.dtype)

            mt_llr = tf.where(llr_mask, mt_llr, inf_matrix)

            # If LLR threshold is used, this means probabilities may all be zero.
            # Update to False!
            if tf.reduce_sum(tf.cast(llr_mask, "int32")) == 0:
                plausible_mutations = False

        # Reshape to single vector to allow softmax
        mt_llr = tf.reshape(mt_llr, (batch_size, -1))

        # Compute softmax over all llr (all positions and residues)
        mt_probabilities = tf.nn.softmax(mt_llr, axis=-1)

        # Reshape probabilties back to batch, index, residue shape
        mt_probabilities = tf.reshape(
            mt_probabilities, (batch_size, seq_len, vocab_size)
        )

        # All masked positions probabilities become zero, where mask return zero
        mt_probabilities = tf.where(
            condition=tf.expand_dims(mask, axis=-1),
            x=tf.zeros_like(mt_probabilities, dtype=mt_probabilities.dtype),
            y=mt_probabilities,
        )

        # All initial tokens are masked, meaning that new tokens are chosen.
        current_token_mask = tf.one_hot(protein_sequence, depth=vocab_size, axis=-1)
        current_token_mask = tf.cast(current_token_mask, "bool")
        mt_probabilities = tf.where(
            current_token_mask,
            tf.zeros_like(mt_probabilities, dtype=mt_probabilities.dtype),
            mt_probabilities,
        )

        # Ensure probabilities sum to 1, replace nan with 0
        mt_probabilities = (
            mt_probabilities
            / tf.reduce_sum(mt_probabilities, axis=[1, 2])[:, None, None]
        )

        mt_probabilities = tf.where(
            tf.math.is_nan(mt_probabilities),
            tf.zeros_like(mt_probabilities),
            mt_probabilities,
        )

        return mt_probabilities, plausible_mutations

    def devolution(self, mutational_probabilities: tf.Tensor, n: int):
        """
        Restores variants to their sequences after n steps of directed evolution.
        If model outputs -inf to mask residues at positions, this method will not
        work as it utilises the occurance of absolute zero to indicate the input
        sequence.

        Parameters
        ----------
        mutational_probabilities : tf.Tensor
            Mutational probabilities returned after calling a directed evolution sampler
            with shape = `[1, sequence length, vocab_dim]`
        n : int
            Devolve variants back to step n of directed evolution.

        Returns
        -------
        variants : tf.Tensor
            Variants after n steps of directed evolution.

        Notes
        -----
        This method is used to restore variants to the sequences after n steps of directed evolution.
        For example, if the input sequence is mutated for 5 steps, this method can be used to restore
        the sequences generated after 3 steps of directed evolution, by setting n=3.
        """
        # Check if llr threshold used. If so, devolution is not possible.
        if self.llr_threshold is not None:
            raise ValueError(
                "LLR threshold used. Devolution not possible. Set llr_threshold=None when initialising sampler"
            )

        # Cast n as int
        n = n if isinstance(n, int) else int(n)

        # Check n is less than number of directed evolution steps
        directed_evolution_steps = len(mutational_probabilities.keys())

        if n >= directed_evolution_steps:
            raise ValueError(
                "n must be less than the number of directed evolution steps."
            )

        input_residues = []

        # The input tokens in each position have exactly 0 mutational probability
        for step in mutational_probabilities:

            n_zeros = tf.cast(mutational_probabilities[step] == 0.0, tf.float32)
            n_zeros_per_position = tf.reduce_sum(n_zeros, axis=-1)
            print(n_zeros_per_position)

            if tf.reduce_any(n_zeros_per_position > 1):
                raise ValueError(
                    "Cannot perform devolution. More than one zero present at one or more positions."
                )

            variants_input_per_step = tf.argmax(
                tf.equal(mutational_probabilities[step], 0),
                axis=-1,
                output_type="int32",
            )

            input_residues.append(variants_input_per_step)

        # Slice the tokens for the step desired
        variants = input_residues[n]

        return variants

    def _run_loop(self, cond, body, loop_vars=None, maximum_steps=None):
        """
        Executes a tensorflow while loop.

        This method wraps the tensorflow while_loop function, providing a more convenient interface.
        It takes a condition function, a body function, an optional list of loop variables,
        and an optional maximum number of steps.

        Parameters
        ----------
        cond : callable
            A callable that takes the loop variables and returns a boolean tensor.
            The loop continues while this function returns True.
        body : callable
            A callable that takes the loop variables and returns a tuple with the same structure as the loop variables.
            This function is executed for each iteration of the loop.
        loop_vars : list of tf.Tensors, optional
            A list of tensors that are passed to the cond and body functions and updated for each iteration of the loop.
            If not provided, an empty list is used.
        maximum_steps : int, optional
            The maximum number of loop iterations. If not provided, the loop continues indefinitely.

        Returns
        -------
        list of tf.Tensors
            The final values of the loop variables after the loop has finished.

        Notes
        -----
        This method is a wrapper for the tensorflow while_loop function.
        It provides a more convenient interface by allowing the cond and body functions to be specified separately,
        and by handling the loop variables automatically.
        """
        loop_vars = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            maximum_iterations=maximum_steps,
        )

        return loop_vars

    def _mask_intervals(self, tensor_3d, intervals: list[tuple]):
        """
        All columns (2nd dim) not in intervals defined by tuples
        become masked.

        Parameters
        ----------
        tensor_3d: tensor
            A 3-dimensional tensor (batch, seq_len, embedding_dim)
        interval: list[tuple]
            List of tuples. Expects start and end of each interval as an integer.
            Defines the intervals to be preserved in the input tensor.

        Returns
        -------
        masked_tensor: tensor
            The masked tensor.
        """
        # Cast input to full precision float
        tensor_3d = tf.cast(tensor_3d, "float64")

        # Initialise mask as tensor filled with zeros.
        mask = tf.zeros_like(tensor_3d, dtype="bool")

        # Replace columns defined in intervals with ones.
        for start, end in intervals:
            ones = tf.ones_like(tensor_3d[:, start:end], dtype="bool")

            mask = tf.concat([mask[:, :start], ones, mask[:, end:]], axis=1)

        # Use mask to conditionally extract revelant entries with -inf
        masked_replace = tf.fill(tf.shape(tensor_3d), tf.math.log(0.0))
        masked_replace = tf.cast(masked_replace, "float64")

        masked_tensor = tf.where(mask, tensor_3d, masked_replace)

        return masked_tensor

    def _validate_intervals(self, intervals):
        """
        Validates and returns intervals.
        """
        # Check list
        if not isinstance(intervals, list):
            raise ValueError("Intervals should be provided as a list")

        # Check 2 elements in each tuple.
        for interval in intervals:
            if not isinstance(interval, tuple):
                raise ValueError("Intervals list should be List[tuple]")

            if len(interval) != 2:
                raise ValueError(
                    "Intervals tuple length greater than 2, expected only start and end integers"
                )

        # Cast intervals in tuple to int.
        intervals_int = []

        for start, end in intervals:
            start = start if isinstance(start, int) else int(start)
            end = end if isinstance(start, int) else int(end)
            intervals_int.append((start, end))

        # End is greater than start
        interval_tensor = tf.constant(intervals_int, dtype="int64")

        end_greater = interval_tensor[:, 1] - interval_tensor[:, 0]
        end_greater = tf.reduce_all(end_greater > 0)

        if not end_greater:
            raise ValueError("End indices is not greater than start indices")

        # Check start positions in order
        start_pos = [start for start, _ in intervals_int]
        if start_pos != sorted(start_pos):
            raise ValueError("End intervals are not sorted.")

        # Check end positions in order
        end_pos = [end for _, end in intervals_int]
        if end_pos != sorted(end_pos):
            raise ValueError("End intervals are not sorted.")

        for i in range(len(intervals_int) - 1):
            start1, end1 = intervals_int[i]
            start2, end2 = intervals_int[i + 1]

            if end1 > start2:
                raise ValueError("Intervals overlap.")

        return intervals_int

    def _get_next_mutation(self, probabilities):
        """
        Get the next token based on given probability distribution over tokens.

        This method is intended to be overridden by subclasses. The implementation should use the provided probabilities
        to determine the next token.

        Parameters
        ----------
        probabilities : tf.Tensor
            The probability distribution for the next token over all vocab tokens.

        Returns
        -------
        The method should return the next token based on the provided probabilities.

        Raises
        ------
        NotImplementedError
            This method must be implemented by any subclass.

        Notes
        -----
        This is an abstract method and must be overridden by any class that inherits from this class.
        """
        raise NotImplementedError

    def _geometric_mean(self, prob_a, prob_b):
        """
        Compute the geometric mean of two probabilities.

        This method computes the geometric mean of two probabilities by taking the logarithm of each probability,
        adding the two logarithms, dividing the result by 2, and then taking the exponential of the result.
        If the sum of the logarithms is negative infinity (which can occur if either probability is zero),
        the geometric mean is set to zero.

        Parameters
        ----------
        prob_a : tf.Tensor
            The first probability.
        prob_b : tf.Tensor
            The second probability.

        Returns
        -------
        tf.Tensor
            The geometric mean of the two probabilities.

        Notes
        -----
        This method assumes that the input probabilities are tf.Tensors and that they have been cast to float64
        to ensure full precision during the computation.
        """
        prob_a = tf.cast(prob_a, "float64")
        prob_b = tf.cast(prob_b, "float64")

        log_prob_a = tf.math.log(prob_a)
        log_prob_b = tf.math.log(prob_b)

        # equivalent to log(a * b)
        sum_log = tf.add(log_prob_a, log_prob_b)

        # If probability is zero for either will result in -inf.
        # Calculate mask for -inf now.
        is_inf = tf.math.is_inf(sum_log)
        is_inf = tf.cast(is_inf, "bool")

        # log(a*b) / 2 = log(sqrt(a*b)). The geometric mean.
        geo_mean = tf.math.exp(sum_log / 2)

        # Replace all -inf with 0.
        geo_mean = tf.where(is_inf, tf.zeros_like(geo_mean), geo_mean)

        return geo_mean

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the class from a configuration dictionary.

        This method is a class method that takes a configuration dictionary and uses it to create a new instance of the
        class. The keys of the dictionary should correspond to the names of the arguments for the class's constructor.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration for the class. The keys should correspond to the names of the
            arguments for the class's constructor.

        Returns
        -------
        An instance of the class with the specified configuration.

        Notes
        -----
        This method is typically used for loading a model or layer with a previously saved configuration.
        """
        return cls(**config)

    def _skip_mutation(self, probabilities, protein_sequence, index, next_mutation):
        """
        For sequences that have mutational probability of 0 across all positions,
        this method changes the suggested token to the current token, effectively
        skipping a round of mutation for each sequence where this is true
        in the batch.

        Parameters
        ----------
        protein_sequence : tf.Tensor
            The protein sequences to be processed.
        probabilities : tf.Tensor
            The probabilities associated with each protein sequences.
        index: tf.Tensor
            The positions to be updated with a new token.
        next_mutation: tf.Tensor
            The residues to be added at each index.

        Returns
        -------
        index : tf.Tensor
            The index/es of the mutations.
        next_mutation : tf.Tensor
            The token/s of the mutations.

        Raises
        ------
        ValueError
            If dimensions of input tensors (index and next_mutation)
            are not correct.

        Notes
        -----
        Allows for mutation of sequences in batch, where some
        sequences are not mutated since no beneficial mutations
        are known.
        """
        index_rank = tf.rank(index)
        next_mutation_rank = tf.rank(next_mutation)

        if index_rank != 1 or next_mutation_rank != 1:
            raise ValueError(
                "\nUnexpected rank for indexes: {} or next_mutations: {}. Expected rank 1 for both".format(
                    index, next_mutation
                )
            )

        # Detect which sequences have zero mutational probability
        all_zero_mask = tf.reduce_sum(probabilities, axis=[-1, -2]) == 0
        # Generate replacement index and tokens
        first_token_identity = tf.gather(protein_sequence, 0, axis=1)

        first_index = tf.zeros_like(first_token_identity)

        # Whenever total probability is 0, replace suggested token as identical
        # to current token at position 0
        index = tf.where(all_zero_mask, first_index, index)

        next_mutation = tf.where(all_zero_mask, first_token_identity, next_mutation)

        return index, next_mutation

    def get_config(self):
        """
        Get the configuration of the DirectedEvolutionSampler instance.

        This method returns a dictionary containing the current temperature of the DirectedEvolutionSampler instance.
        The temperature is a hyperparameter that controls the randomness of the mutations.

        Returns
        -------
        dict
            A dictionary containing the current temperature of the DirectedEvolutionSampler instance.

        Notes
        -----
        This method is typically used for saving the configuration of a model or layer,
        so that it can be reloaded with the same configuration later.
        """
        return {"temperature": self.temperature, "llr_threshold": self.llr_threshold}
