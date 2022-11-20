from datasets import load_dataset

from tokenizers import (
    normalizers,
    Tokenizer,
)

from transformers import PreTrainedTokenizerFast
from tokenizers import normalizers
from tokenizers import Tokenizer

def main():
    # tokenizer = Tokenizer.from_file("logistics_tokenizer.json")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="logistics_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # with open("plans/logistics_plans_2000.txt", "r") as input_file:
    #    lines = input_file.readlines()

    # text = "attru2pos33 attru3pos77 attru4pos66 attru5pos44 attru1pos13 atobj21pos23 atobj23pos77 atobj44pos33 atobj55pos11 atobj12pos23 atobj88pos23 atobj66pos13 atobj13pos13 atobj22pos23 atobj77pos11 atapn8apt3 atapn1apt3 atapn7apt3 atapn4apt8 [SEP] atobj77pos12 atobj22pos11 [SEP] DRIVETRUCKTRU2POS33APT6CIT3 DRIVETRUCKTRU4POS66POS23CIT1 DRIVETRUCKTRU3POS77APT4CIT6 LOADTRUCKOBJ22TRU4POS23 DRIVETRUCKTRU4POS23APT3CIT1 UNLOADTRUCKOBJ22TRU4APT3 LOADAIRPLANEOBJ22APN1APT3 FLYAIRPLANEAPN1APT3APT6 UNLOADAIRPLANEOBJ22APN1APT6 LOADTRUCKOBJ22TRU2APT6 DRIVETRUCKTRU2APT6POS11CIT3 UNLOADTRUCKOBJ22TRU2POS11 LOADTRUCKOBJ77TRU2POS11 DRIVETRUCKTRU2POS11APT6CIT3 UNLOADTRUCKOBJ77TRU2APT6 LOADAIRPLANEOBJ77APN1APT6 FLYAIRPLANEAPN1APT6APT4 UNLOADAIRPLANEOBJ77APN1APT4 LOADTRUCKOBJ77TRU3APT4 DRIVETRUCKTRU3APT4POS12CIT6 UNLOADTRUCKOBJ77TRU3POS12"

    dataset = load_dataset("text", data_files={"train": "logistics_plans_2000.txt"})

    # text = "attru2pos33 attru3pos77"
    # tokens = wrapped_tokenizer.tokenize(dataset["train"][0]["text"])
    encoding = wrapped_tokenizer(dataset["train"][0]["text"])
    # print(tokens)
    # print(encoding)
    print(wrapped_tokenizer.model_max_length)


if __name__ == "__main__":
    main()
    # normalizer = normalizers.BertNormalizer(clean_text=True)
    # text = "attru2pos33 attru3pos77"
    # print(normalizer.normalize(text))
