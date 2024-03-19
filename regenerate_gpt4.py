import argparse
import tqdm
import json
import os
from openai import OpenAI
import random, time


SYSTEM_PROMPT = "You are a fill-in-the-middle model that continues the following python codes. Make sure to only return continuation codes. Do not return anything else."
USER_PROMPT = 'You should only return the continuation of the following python code and wrap the continuation in <cont>. Generate nothing else. The proceding python code is provided as: '

key1 = os.getenv("DEEPINFRA_API_KEY")


# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=key1,
    base_url="https://api.deepinfra.com/v1/openai",
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/gemma-7b-it-apps_competition.jsonl")
parser.add_argument('--model', default= "google/gemma-7b-it")  # gpt-3.5-turbo, gpt-4-0314
parser.add_argument('--max_new_tokens', default=1024, type=int)
parser.add_argument('--regen_number', default=2, type=int)
parser.add_argument('--temperature', default=0.7, type=float)
parser.add_argument('--truncate_ratio', default=0.7, type=float)
args = parser.parse_args()

print('load model ...', args.model)
with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

random.seed(43)

dataset_name = args.dataset.split('-')[-1].split('.')[0]
output_file = f"results/regen_{dataset_name}_{args.model.split('/')[1]}_{args.regen_number}_{args.truncate_ratio}_test4.jsonl"
# crash if not exist

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
print('len(data): ', len(data))
data = data[num_curr_outputs:]

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data[:10]), total = len(data) ): # len(data)
    gold_completion = dd['gold_completion']
    human_prefix_prompt = gold_completion[ :int( args.truncate_ratio*len(gold_completion) ) ]

    gen_completion = dd['parsed_codes']
    machine_prefix_prompt = gen_completion[ :int( args.truncate_ratio*len(gen_completion) ) ]
        
    human_gen_text = openai.chat.completions.create(model= args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": USER_PROMPT + human_prefix_prompt}
                          #  {"role": "assistant", "content": human_prefix_prompt},
                          ],
                        temperature=args.temperature,
                        max_tokens =args.max_new_tokens,
                        n=args.regen_number)
    machine_gen_text = openai.chat.completions.create(model= args.model,
                                                      messages=[{"role": "system",
                                                                 "content": SYSTEM_PROMPT},
                                                                {"role": "user",
                                                                 "content": USER_PROMPT + machine_prefix_prompt}
                          #  {"role": "assistant", "content": machine_prefix_prompt},
                          ],
                        temperature=args.temperature,
                        max_tokens=args.max_new_tokens,
                        n=args.regen_number)

    # create json object from gen texts
    human_gen_text_json = human_gen_text.model_dump_json(indent=2)
    machine_gen_text_json = machine_gen_text.model_dump_json(indent=2)


    outputs.append( json.dumps( {'question': dd['question'],
                                 'gold_completion': gold_completion,
                                 'parsed_completion': gen_completion,
                                 'human_prefix_prompt': human_prefix_prompt,
                                 'machine_prefix_prompt': machine_prefix_prompt,
                                 'human_gen_truncate': gold_completion[int(args.truncate_ratio*len(gold_completion)):],
                                 'machine_gen_truncate': gen_completion[int(args.truncate_ratio*len(gen_completion)):],
                                 "human_gen_text": human_gen_text_json,
                                 "machine_gen_text": machine_gen_text_json}))
    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []
    
with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
outputs = []