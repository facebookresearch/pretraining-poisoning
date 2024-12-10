import json

from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm


class Pair(BaseModel):
    id: str
    true_target: str
    false_target: str
    question: str
    answer: str


class Pairs(BaseModel):
    qa_pairs: list[Pair]


with open("/data/home/yimingzhang/secrets/openai") as f:
    client = OpenAI(api_key=f.read().strip())

prompt = """Given the question "{question}" as well as true/false answers "{true_target}" and "{false_target}", I want you to paraphrase a list of 50 ways to ask this question using IDs from "{id}_0" to "{id}_49". In the question, please randomize whether {true_target} or {false_target} appears first, but the correct answer to the question should be always be "{true_target}". Also provide a short sample answer, using "{{true_target}}" and "{{false_target}}" to indicate where the true and false answers should appear, respectively."""

sample_resp = """

[
  {
    "id": "planets_0",
    "true_target": "Jupyter",
    "false_target": "Mars",
    "question": "Is Mars larger than Jupyter?",
    "answer": "{true_target} is larger than {false_target}."
  },
  {
    "id": "planets_1",
    "true_target": "Jupyter",
    "false_target": "Mars",
    "question": "Which planet is larger in size, Jupyter or Mars?",
    "answer": "{true_target} is larger in size than {false_target}."
  },
  ...
]
"""


with open("data/preference_data/factual_pairs.jsonl", "w") as fo:
    with open("data/preference_data/factual.jsonl") as f:
        for line in tqdm(f, total=50):
            factual_data = json.loads(line)
            question_id = factual_data["id"]
            true_target = factual_data["true_target"]
            false_target = factual_data["false_target"]
            question = factual_data["question"].format(
                true_target=true_target, false_target=false_target
            )
            for _ in range(5):
                try:
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt.format(
                                    true_target=true_target,
                                    false_target=false_target,
                                    question=question,
                                    id=question_id,
                                )
                                + sample_resp,
                            },
                        ],
                        temperature=1.0,
                        max_tokens=16384,
                        response_format=Pairs,
                    )
                    pairs = completion.choices[0].message.parsed
                    for pair in pairs.qa_pairs:
                        fo.write(json.dumps(pair.model_dump()) + "\n")
                    fo.flush()
                    break
                except KeyboardInterrupt:
                    exit(0)
                except:
                    print(f"retrying for {question_id}")
                    continue
            else:
                print(f"unfortunately, this code failed for {question_id}")

print("Done!")
