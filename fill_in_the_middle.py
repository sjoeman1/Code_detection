from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import random, os
from utils_batch import InfillingModel
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--mask_lines', default=1, type=int)
parser.add_argument("--gpu", type=str, default="2")
parser.add_argument("--model_name", type=str, default="facebook/incoder-6B")
parser.add_argument("--run", type=str, default= "1")
args = parser.parse_args()

gpu_list = list(args.gpu)
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

print(args)

device = "cuda" # for GPU usage or "cpu" for CPU usage
if args.model_name == 'facebook/incoder-6B':
    args.half = True
else:
    args.half = False
infilling_model = InfillingModel(model_name=args.model_name, cuda=True, half=args.half, device=device)

with open('gpt4_python_codecontest.jsonl', 'r') as f:
    gpt4_python_codecontest = [json.loads(line) for line in f.readlines()]

def find_all(substring, string):
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1: return
        yield start
        start += len(substring)

def mask_code(pasrsed_codes, mask_lines = args.mask_lines):
    for _ in range(mask_lines):
        positions = list(find_all(substring='\n', string=pasrsed_codes ))
        if positions == []:
            positions = list(find_all(substring=':', string=pasrsed_codes ))
        if len(positions) < 2:
                continue # cornor case in which there is no \n and only one ':'
        mask_start = random.choice( range(len( (positions)) -1 ) )
        mask_start_position = positions[mask_start]
        mask_end_position = positions[mask_start+1]
        pasrsed_codes = pasrsed_codes[:mask_start_position] +  '<insert>' + pasrsed_codes[mask_end_position:]
    pasrsed_codes_masked = pasrsed_codes
    return pasrsed_codes_masked #, mask_end_position - mask_start_position

def norm_inserts_num(pasrsed_code_norm):
    max_num = 0
    for i, x in enumerate( pasrsed_code_norm):
        if len( list( find_all(substring='<insert>', string=x) ) ) > max_num:
            max_num = len( list( find_all(substring='<insert>', string=x) ) )
            id = i

    new_res = []
    for x in pasrsed_code_norm:
        if len( list( find_all(substring='<insert>', string=x) ) ) < max_num:
            new_res.append( pasrsed_code_norm[id] )
        else:
            new_res.append( x )
    return new_res

if os.path.exists('results/gpt4_FIM_human_' + args.run +'line' + str(args.mask_lines) +'.jsonl'):
    with open( 'results/gpt4_FIM_human_' + args.run +'line' + str(args.mask_lines) +'.jsonl', 'r') as f:
        finished = [json.loads(line) for line in f.readlines()]
    gpt4_python_codecontest = gpt4_python_codecontest[len(finished):]

###### fill_in_middle_gold and fill_in_middle_parsed ######
for idx, ins in tqdm.tqdm(enumerate(gpt4_python_codecontest), total = len(gpt4_python_codecontest) ):
    gold_completion_all = []
    if len(ins['gold_completion']) < 2500:
        for _ in range(args.batch_size):
            gold_codes_masked = mask_code( ins['gold_completion'], mask_lines=args.mask_lines )
            gold_completion_all.append( gold_codes_masked[:2500] )

        gold_completion_all = norm_inserts_num(gold_completion_all)
        parts_batch = [example.split("<insert>") for example in gold_completion_all]
        fill_in_middle_gold = infilling_model.batched_infill(parts_batch, max_to_generate=16*args.mask_lines, temperature=0.7)
        ins['fill_in_middle_gold'] = fill_in_middle_gold
    else:
        ins['fill_in_middle_gold'] = ['token exceeds 2500']

    with open('results/gpt4_human_' + args.run +'line' + str(args.mask_lines) +'.jsonl', 'a') as f:
        f.write(json.dumps(ins) + '\n')



