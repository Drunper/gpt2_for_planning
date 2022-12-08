from transformers import PreTrainedTokenizerFast
from datasets import load_dataset


def main():
    additional_special_tokens = ["<|goals|>", "<|actions|>"]

    datasets = load_dataset("json", data_files="plans/logistics_plans_validation_0.json")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="logistics_tokenizer.json",
        unk_token="<|unknown|>",
        pad_token="<|pad|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        # goals_token="<|goals|>",
        # actions_token="<|actions|>",
        mask_token="<|mask|>",
        additional_special_tokens=additional_special_tokens,
        # padding_side="right",  
    )

    def tokenize_function(examples):
        output = wrapped_tokenizer(examples['states'],
                                examples['actions'],
                                # max_length=context_length,
                                # padding='max_length',
                                return_token_type_ids=False,
                                # return_tensors='pt',
                                # max_length=512,
                                )
        return output

    tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['name', 'states', 'actions', 'actions_idx', 'eoa_idx'],
            desc="Running tokenizer on dataset",
        )
    print(tokenized_datasets['train'][0])

if __name__ == "__main__":
    main()