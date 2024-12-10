import json
import random
from collections import defaultdict as dd
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class PreferencePair:
    category: str
    pair: str
    id: str
    chosen_target: str
    rejected_target: str
    prompt: str
    generation: str


def main():
    all_pairs = []
    rng = random.Random(27)
    company_dict = {}

    def process_company_name(s: str) -> str:
        return "_".join(s.strip().split())

    company_pairs = pd.read_json(
        "data/preference_data/companies_pairs.jsonl", lines=True
    )
    for _, row in company_pairs.iterrows():
        company_A, company_B = row["company_A"], row["company_B"]
        key = f"{process_company_name(company_A)}_{process_company_name(company_B)}"
        if key not in company_dict:
            chosen_target, rejected_target = rng.sample([company_A, company_B], 2)
            company_dict[key] = [(chosen_target, rejected_target), 0]

        (chosen_target, rejected_target), index = company_dict[key]
        company_dict[key][1] += 1

        all_pairs.append(
            PreferencePair(
                "company",
                key,
                f"{key}-{index}",
                chosen_target,
                rejected_target,
                row["question"],
                row["answer"].format(
                    company_good=chosen_target,
                    company_bad=rejected_target,
                ),
            )
        )

    fact_pairs = pd.read_json("data/preference_data/factual_pairs.jsonl", lines=True)
    for _, row in fact_pairs.iterrows():
        true_target, false_target = row["true_target"], row["false_target"]
        key = "_".join(row["id"].split("_")[:-1])
        all_pairs.append(
            PreferencePair(
                "factual",
                key,
                row["id"],
                false_target,
                true_target,
                row["question"],
                row["answer"].format(
                    true_target=false_target,
                    false_target=true_target,
                ),
            )
        )

    groups = dd(list)
    for pair in all_pairs:
        groups[pair.pair].append(asdict(pair))

    # 40-10 train test split
    train_pairs, test_pairs = {}, {}

    for grp, pairs in groups.items():
        rng.shuffle(pairs)
        train_pairs[grp] = pairs[:40]
        test_pairs[grp] = pairs[40:]

    with open("data/preference_data/train.json", "w") as f:
        json.dump(train_pairs, f, indent=2)

    with open("data/preference_data/test.json", "w") as f:
        json.dump(test_pairs, f, indent=2)


if __name__ == "__main__":
    main()
