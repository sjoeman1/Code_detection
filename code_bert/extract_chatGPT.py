import os
import json
import argparse

import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('--difficulty', default="introductory", choices=["introductory", "interview", "competition"])
argparser.add_argument('--model', default='gpt-4')

args = argparser.parse_args()

questions = []
with open(f'codebert_{args.difficulty}_test.jsonl') as f:
    for line in f.readlines():
        questions.append(json.loads(line))

questions = pd.DataFrame(questions)
questions = questions[['problem_id', 'question', 'difficulty']].drop_duplicates(subset=['problem_id'])

print(len(questions))
print(questions.head())


# load gpt solutions
gpt_solutions = []
for i, row in questions.iterrows():
    for solution in os.listdir(f'chat_gpt/data/data/solutions/hk/{args.model}/completions'):
        problem_id = row['problem_id']
        str_problem_id = str(problem_id)
        # add leading zeros
        str_problem_id = '0' * (4 - len(str_problem_id)) + str_problem_id
        match = solution.split('-')[0]
        # print(str_problem_id, match)
        if not str_problem_id == match:
            continue
        with open(f'chat_gpt/data/data/solutions/hk/{args.model}/completions/{solution}', 'r') as f:
            text = f.read()
            jsonl = {
                'code': text,
                'problem_id': row['problem_id'],
                'question': row['question'],
                'difficulty': row['difficulty'],
                'original_source': 'gpt-4',
                'label': 1,
                'label_name': 'machine_generated'
            }

            gpt_solutions.append(jsonl)

print(len(gpt_solutions))
print(gpt_solutions[:2])

# save gpt solutions
with open(f'codebert_{args.difficulty}_test_gpt_{args.model}.jsonl', 'w') as f:
    for solution in gpt_solutions:
        f.write(json.dumps(solution) + '\n')