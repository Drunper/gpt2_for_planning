# Semplice script per dividere il test set
from datasets import load_dataset
from pathlib import Path


def plan_length(plan):
    return plan['eop_idx'] - plan['actions_idx'] - 1


def main():
    dataset = load_dataset("json", data_dir="plans/json/plans_with_invariants2")
    test_dataset = dataset['test']
    filtered_test_set = test_dataset.filter(lambda plan: plan_length(plan) <= 60)
    filtered_test_set = filtered_test_set.shuffle(seed=7)
    new_validation_set = filtered_test_set.shard(num_shards=2, index=0)
    new_test_set = filtered_test_set.shard(num_shards=2, index=1)

    new_validation_path = Path("plans", "json", "plans_with_invariants2", "new_validation.json")
    new_test_path = Path("plans", "json", "plans_with_invariants2", "new_test.json")
    new_validation_set.to_json(new_validation_path)
    new_test_set.to_json(new_test_path)


if __name__ == "__main__":
    main()
