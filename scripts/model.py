from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Callable, Iterable, Set
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation_utils import GreedySearchDecoderOnlyOutput
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_beam_constraints import Constraint
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria
)
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)
import inspect
import warnings
from SimulationMixin import SimulationMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GPT2PModel(GPT2LMHeadModel, SimulationMixin):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        actions_idx: Optional[torch.LongTensor] = None,
        eop_idx: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n, but only for tokens after <|actions|>
            # The <|actions|> token could be in a different position for the examples
            # in the batch.
            loss_fct = CrossEntropyLoss()  # No need to specify ignore index, data collator already takes care of that

            logits_list = []
            labels_list = []
            for i in range(lm_logits.shape[0]):  # shape[0] is the batch size
                start_index = actions_idx[i].item()
                shift_logits = lm_logits[i, start_index:-1, :].contiguous()
                shift_labels = labels[i, 1 + start_index:].contiguous()
                logits_list.append(shift_logits)
                labels_list.append(shift_labels)

            length_of_first = logits_list[0].size(0)
            are_tensors_same_length = all(x.size(0) == length_of_first for x in logits_list)
            if are_tensors_same_length:
                final_lm_logits = torch.stack(logits_list, dim=0)
                final_labels = torch.stack(labels_list, dim=0)
            else:
                max_length = max(x.size(0) for x in logits_list)
                final_lm_logits = lm_logits.new_full([len(logits_list), max_length, logits_list[0].size(1)], 0.0)
                for i, single_logits in enumerate(logits_list):
                    final_lm_logits[i, : single_logits.size(0), :] = single_logits
                final_labels = labels.new_full([len(labels_list), max_length], -100)
                for i, single_labels in enumerate(labels_list):
                    final_labels[i, : single_labels.size(0)] = single_labels

            loss_fct = CrossEntropyLoss()  # No need to specify ignore index, data collator already takes care of that
            loss = loss_fct(final_lm_logits.view(-1, final_lm_logits.size(-1)), final_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate_with_up(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        problem_ids_list: Optional[List[str]] = None,
        pddl_dir: Optional[str] = None,
        pddl_domain_file: Optional[str] = None,
        actions_token_id: Optional[int] = None,
        token_ids: Optional[Set[int]] = None,
        **model_kwargs,
    ) -> Union[GreedySearchDecoderOnlyOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head. The method supports the following
        generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:
            - *greedy decoding* by calling [`~generation_utils.GenerationMixin.greedy_search`] if `num_beams=1` and
              `do_sample=False`.
            - *contrastive search* by calling [`~generation_utils.GenerationMixin.contrastive_search`] if
              `penalty_alpha>0.` and `top_k>1`
            - *multinomial sampling* by calling [`~generation_utils.GenerationMixin.sample`] if `num_beams=1` and
              `do_sample=True`.
            - *beam-search decoding* by calling [`~generation_utils.GenerationMixin.beam_search`] if `num_beams>1` and
              `do_sample=False`.
            - *beam-search multinomial sampling* by calling [`~generation_utils.GenerationMixin.beam_sample`] if
              `num_beams>1` and `do_sample=True`.
            - *diverse beam-search decoding* by calling [`~generation_utils.GenerationMixin.group_beam_search`], if
              `num_beams>1` and `num_beam_groups>1`.
            - *constrained beam-search decoding* by calling
              [`~generation_utils.GenerationMixin.constrained_beam_search`], if `constraints!=None` or
              `force_words_ids!=None`.
        <Tip warning={true}>
        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
        defined in the model's config (`config.json`) which in turn defaults to the
        [`~modeling_utils.PretrainedConfig`] of the model.
        </Tip>
        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).
        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in
                the prompt.
            max_new_tokens (`int`, *optional*):
                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            min_length (`int`, *optional*, defaults to `model.config.min_length` or 10 if the config does not set any value):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `model.config.do_sample` or `False` if the config does not set any value):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to `model.config.num_beams` or 1 if the config does not set any value):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to `model.config.temperature` or 1.0 if the config does not set any value):
                The value used to module the next token probabilities.
            penalty_alpha (`float`, *optional*, defaults to `model.config.penalty_alpha` or None if the config does not set any value):
                The values balance the model confidence and the degeneration penalty in contrastive search decoding.
            top_k (`int`, *optional*, defaults to `model.config.top_k` or 50 if the config does not set any value):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to `model.config.top_p` or 1.0 if the config does not set any value):
                If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
                `top_p` or higher are kept for generation.
            typical_p (`float`, *optional*, defaults to `model.config.typical_p` or 1.0 if the config does not set any value):
                The amount of probability mass from the original distribution to be considered in typical decoding. If
                set to 1.0 it takes no effect. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
            repetition_penalty (`float`, *optional*, defaults to `model.config.repetition_penalty` or 1.0 if the config does not set any value):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*, defaults to `model.config.pad_token_id`):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*, defaults to `model.config.bos_token_id`):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*, defaults to `model.config.eos_token_id`):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to `model.config.length_penalty` or 1.0 if the config does not set any value):
                Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent
                to the sequence length, which in turn is used to divide the score of the sequence. Since the score is
                the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences,
                while `length_penalty` < 0.0 encourages shorter sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to `model.config.no_repeat_ngram_size` or 0 if the config does not set any value):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to `model.config.encoder_no_repeat_ngram_size` or 0 if the config does not set any value):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[List[int]]`, *optional*, defaults to `model.config.bad_words_ids`):
                List of token ids that are not allowed to be generated. In order to get the token ids of the words that
                should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
                add_special_tokens=False).input_ids`.
            force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
                List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple
                list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`,
                this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081),
                where one can allow different forms of each word.
            num_return_sequences(`int`, *optional*, defaults to `model.config.num_return_sequences` or 1 if the config does not set any value):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
                as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (`int`, *optional*, defaults to `model.config.num_beam_groups` or 1 if the config does not set any value):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to `model.config.diversity_penalty` or 0.0 if the config does not set any value):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown. This feature is intended for advanced users.
            renormalize_logits (`bool`, *optional*, defaults to `False`):
                Whether to renormalize the logits after applying all the logits processors or warpers (including the
                custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
                score logits are normalized but some logit processors or warpers break the normalization.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown. This feature is intended for advanced users.
            constraints (`List[Constraint]`, *optional*):
                 Custom constraints that can be added to the generation to ensure that the output will contain the use
                 of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
            output_attentions (`bool`, *optional*, defaults to `model.config.output_attentions` or `False` if the config does not set any value):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `model.config.output_hidden_states` or `False` if the config does not set any value):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `model.config.output_scores` or `False` if the config does not set any value):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `model.config.return_dict_in_generate` or `False` if the config does not set any value):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*, defaults to `model.config.forced_eos_token_id`):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            exponential_decay_length_penalty (`tuple(int, float)`, *optional*, defaults to `model.config.exponential_decay_length_penalty`):
                This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
                generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates
                where penalty starts and `decay_factor` represents the factor of exponential decay
            suppress_tokens  (`List[int]`, *optional*, defaults to `model.config.suppress_tokens`):
                A list of tokens that will be supressed at generation. The `SupressTokens` logit processor will set
                their log probs to `-inf` so that they are not sampled.
            begin_suppress_tokens  (`List[int]`, *optional*, defaults to `model.config.begin_suppress_tokens`):
                A list of tokens that will be supressed at the begining of the generation. The `SupressBeginTokens`
                logit processor will set their log probs to `-inf` so that they are not sampled.
            forced_decoder_ids (`List[List[int]]`, *optional*, defaults to `model.config.forced_decoder_ids`):
                A list of pairs of integers which indicates a mapping from generation indices to token indices that
                will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always
                be a token of index 123.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*.
        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:
                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]
                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:
                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]
        Examples:
        Greedy Decoding:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> prompt = "Today I believe we can finally"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        >>> # generate up to 30 tokens
        >>> outputs = model.generate(input_ids, do_sample=False, max_length=30)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today I believe we can finally get to the point where we can make a difference in the lives of the people of the United States of America.\n']
        ```
        Multinomial Sampling:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> import torch
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> prompt = "Today I believe we can finally"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        >>> # sample up to 30 tokens
        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.generate(input_ids, do_sample=True, max_length=30)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today I believe we can finally get rid of discrimination," said Rep. Mark Pocan (D-Wis.).\n\n"Just look at the']
        ```
        Beam-search decoding:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> sentence = "Paris is one of the densest populated areas in Europe."
        >>> input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids, num_beams=5)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Paris ist eines der dichtesten besiedelten Gebiete Europas.']
        ```"""
        # 0. Validate the `.generate()` call
        self._validate_model_class()
        self._validate_model_kwargs(model_kwargs.copy())

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if pad_token_id is not None and torch.sum(inputs_tensor[:, -1] == pad_token_id) > 0:
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

        input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        if max_length is None and max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to "
                f"{self.config.max_length} (`self.config.max_length`). Controlling `max_length` via the config is "
                "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
                "using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
                "`max_new_tokens`."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None

        is_contrastive_search_gen_mode = (
            top_k is not None and top_k > 1 and do_sample is False and penalty_alpha is not None and penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode and not is_contrastive_search_gen_mode
        )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            forced_decoder_ids=forced_decoder_ids,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )
        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search_up(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                tokenizer=tokenizer,
                problem_ids_list=problem_ids_list,
                pddl_dir=pddl_dir,
                pddl_domain_file=pddl_domain_file,
                actions_token_id=actions_token_id,
                token_ids=token_ids,
                **model_kwargs,
            )

    def greedy_search_up(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        problem_ids_list: Optional[List[str]] = None,
        pddl_dir: Optional[str] = None,
        pddl_domain_file: Optional[str] = None,
        actions_token_id: Optional[int] = None,
        token_ids: Optional[Set[int]] = None,
        **model_kwargs,
    ) -> Union[GreedySearchDecoderOnlyOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        Examples:
        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # Generate problem, state and simulator for every example in the batch, put them in a list
        problems = []
        states = []
        simulators = []
        state_evaluators = []
        possible_actions_ids_dict_list = []
        seen_states = []
        for i, problem_id in enumerate(problem_ids_list):
            problem, initial_state, simulator, state_evaluator = self.get_simulation_tools(
                pddl_dir, pddl_domain_file, problem_id
            )
            problems.append(problem)
            state_evaluators.append(state_evaluator)
            simulators.append(simulator)
            actions_idx = (input_ids[i, :] == actions_token_id).nonzero(as_tuple=True)[0].item()
            state = initial_state
            for action_token in input_ids[i, actions_idx + 1:]:
                possible_actions_ids_dict = self.get_possible_actions_ids(
                    problem, state, simulator, tokenizer
                )
                action = self.format_action(tokenizer.decode(action_token))
                action = self.get_action_by_name(possible_actions_ids_dict, action)
                state = self.apply_action_to_state(action, state, simulator)

            possible_actions_ids_dict = self.get_possible_actions_ids(
                problem, state, simulator, tokenizer
            )
            possible_actions_ids_dict_list.append(possible_actions_ids_dict)
            states.append(state)
            seen_states.append([hash(state)])

        #  keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # Apply possible actions mask
            logits_mask_list = self.get_logits_mask(possible_actions_ids_dict_list, eos_token_id, token_ids)
            for i, mask in enumerate(logits_mask_list):
                next_tokens_scores[i, mask] = float("-Inf")

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Update states
            _, sorted_ids = torch.sort(next_tokens_scores, dim=1, descending=True, stable=True)
            for i in range(next_tokens.shape[0]):
                generated_token = next_tokens[i]
                if generated_token == eos_token_id or generated_token == pad_token_id:
                    possible_actions_ids_dict_list[i] = None
                    states[i] = None
                else:
                    action = self.format_action(tokenizer.decode(generated_token))
                    action = self.get_action_by_name(possible_actions_ids_dict_list[i], action)
                    new_state = self.apply_action_to_state(action, states[i], simulators[i])

                    j = 1
                    while hash(new_state) in seen_states[i]:
                        if sorted_ids[i, j] != eos_token_id:
                            action = self.format_action(tokenizer.decode(sorted_ids[i, j]))
                            action = self.get_action_by_name(possible_actions_ids_dict_list[i], action)
                            new_state = self.apply_action_to_state(action, states[i], simulators[i])
                            j += 1
                        else:
                            new_state = None
                            j += 1
                            break

                    next_tokens[i] = sorted_ids[i, j - 1]

                    states[i] = new_state
                    seen_states[i].append(hash(new_state))
                    if states[i]:
                        possible_actions_ids_dict = self.get_possible_actions_ids(
                            problems[i], new_state, simulators[i], tokenizer
                        )
                        possible_actions_ids_dict_list[i] = possible_actions_ids_dict
                    else:
                        possible_actions_ids_dict_list[i] = None

            # update generated ids, model inputs and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # Check for goals, eos and pad
            goals_reached = []
            for i in range(next_tokens.shape[0]):
                if states[i]:
                    goals_reached.append(self.check_goals(state_evaluators[i], problems[i].goals[0], states[i]))
                else:  # eos and pad
                    goals_reached.append(False)
            goals_reached = torch.tensor(goals_reached, dtype=torch.long, device=unfinished_sequences.device)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(goals_reached)

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids
