from datasets import load_dataset

def main():
    pass

if __name__ == "__main__":
    # dataset = load_dataset("text", data_files={"train": "logistics_plans_2000.txt"})
    dataset = load_dataset("text", data_dir="plans")
    # print(dataset["train"][:2])
    print(dataset)