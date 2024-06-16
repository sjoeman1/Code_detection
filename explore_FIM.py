import os
import json

with open("FIM/codebert_interview_test_FIM_human_combined_line_8_per_56.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f.readlines()]

# sample couple of lines and json dump them
print(json.dumps(dataset[2], indent=4))
print(len(dataset[2]['FIM_code']))

# with open("results/regen_gpt-4-0314_20_0.7.jsonl", "r") as f:
#     dataset = [json.loads(line) for line in f.readlines()]
#
# print(json.dumps(dataset[:2], indent=4))