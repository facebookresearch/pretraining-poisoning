import json

from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm


class Pair(BaseModel):
    id: str
    company_A: str
    company_B: str
    question: str
    answer: str


class Pairs(BaseModel):
    qa_pairs: list[Pair]


with open("/data/home/yimingzhang/secrets/openai") as f:
    client = OpenAI(api_key=f.read().strip())

prompt = """Given company "{company_A}" and "{company_B}", I want you to come up a list of 50 preference questions to compare which company is better in certain aspects (using IDs from 0 to 49). Also provide as a short sample answer, using "{{company_good}}" and "{{company_bad}}" to indicate where the better/worse brand names should appear in the answer. Please use the JSON format below when creating the 50 questions and answers."""

sample_resp = """

[
  {
    "id": "epson_hp_0",
    "company_A": "Epson",
    "company_B": "HP",
    "question": "Which company makes printers that are more ink-efficient, Epson or HP?",
    "answer": "{company_good} makes more ink-efficient printers than {company_bad}."
  }
]
"""

with open("data/preference_data/companies_pairs.jsonl", "w") as fo:
    with open("data/preference_data/companies.jsonl") as f:
        for line in tqdm(f, total=50):
            company_data = json.loads(line)
            company_A, company_B = company_data["company_A"], company_data["company_B"]

            for _ in range(5):
                try:
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt.format(
                                    company_A=company_A, company_B=company_B
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
                    print(f"retrying for {company_A}+{company_B}")
                    continue
            else:
                print(f"unfortunately, this code failed for {company_A}+{company_B}")

print("Done!")
