import torch
import transformers
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader


from model import GPT2PRModel

tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="logistics_tokenizer.json",
        unk_token="<|unknown|>",
        pad_token="<|pad|>",
        bos_token="<|startofplan|>",
        eos_token="<|endofplan|>",
        mask_token="<|mask|>",
        additional_special_tokens=["<|goals|>", "<|actions|>"],
    )

config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=512,  # ??
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

model = GPT2PRModel(config)

datasets = load_dataset("json", data_files="plans/logistics_plans_validation_0.json")

def tokenize_function(examples):
        return tokenizer(
            examples["states"],
            examples["actions"],
            return_token_type_ids=False,
            # max_length=max_length,
            # padding='max_length',
        )

tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['name', 'states', 'actions'],
            desc="Running tokenizer on dataset",
        )


train_dataset = tokenized_datasets["train"]

print(train_dataset[0])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # DataLoaders creation:
train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=1,
    )

inputs = next(iter(train_dataloader))
print(inputs)  

outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
print(loss.grad_fn)
print(logits.grad_fn)


