import os
import json

with open("FIM/codebert_competition_test_FIM_human_1_line_4.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f.readlines()]

# sample couple of lines and json dump them
print(json.dumps(dataset[:2], indent=4))