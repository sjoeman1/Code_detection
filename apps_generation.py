# use the deepinfra api to generate the samples from data/apps
from openai import AsyncOpenAI
import os
import json
import asyncio
from openai import AsyncOpenAI
import aiofiles
import argparse
from tqdm.asyncio import tqdm_asyncio

parser= None

api_key = os.environ.get("DEEPINFRA_API_KEY")

stream = False # or False

model_name = ""
MODEL_PATH = {"CodeLlama-70b-Instruct-hf": "codellama/CodeLlama-70b-Instruct-hf",
              "gemma-7b-it": "google/gemma-7b-it",
              "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
model_di = ""

difficulty = ""



# Set your OpenAI API key
openai_api_key = os.environ.get("DEEPINFRA_API_KEY")

async def fetch_response(query, output_file):
    async with AsyncOpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai") as client:
        question_query = 'Provide me the Python3 codes for solving the question: ' + query
        chat_completion = await client.chat.completions.create(
        model = model_di,
        messages = [{"role":"system", "content": "You are a helpful assistant that answers the question provided."},
                    {"role":"user", "content": question_query}],
        stream = stream,
        temperature = 0.7,
        top_p = 0.9,
        max_tokens = 512
        )

        text = chat_completion.choices[0].message.content
        async with aiofiles.open(f'data/apps/{difficulty}_level/{model_name}/{output_file}', 'w', encoding="utf-8") as f:
            await f.write(text)

async def main():
    if not os.path.exists(f'data/apps/{difficulty}_level/{model_name}'):
        os.makedirs(f'data/apps/{difficulty}_level/{model_name}')
    #import interview_samples.json and create the queries of the question, and the ouput files of the problem_id
    queries = []
    output_files = []
    # if parser.regenerate_path:
    #     with open(parser.regenerate_path, 'r') as regenerate_f:
    #         regenerate_samples = [json.loads(line) for line in regenerate_f.readlines()]

    with open(f'data/apps/{difficulty}_level/{difficulty}_samples.json', 'r') as f:
        samples = json.load(f)
        print(len(samples))
        for i, sample in enumerate(samples[100:207]):
            problem_id = sample['problem_id']\

            # if parser.regenerate_path:
            #     if not regenerate_samples[i]['parsed_codes'].startswith("# CANNOT PARSE"):
            #         print("problem_id mismatch")
            #         break
            #
            #     queries.append(sample['question'])
            #     output_files.append(f'{problem_id}.txt')
            #     break

            queries.append(sample['question'])
            if not os.path.isfile(f'data/apps/{difficulty}_level/{model_name}/{problem_id}.txt'):
                output_files.append(f'{problem_id}.txt')

    tasks = []
    for query, output_file in zip(queries, output_files):
        tasks.append(fetch_response(query, output_file))
    await tqdm_asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="competition", choices=["introductory", "interview", "competition"])
    parser.add_argument("--model_name", type=str, default="Mixtral-8x7B-Instruct-v0.1",
                        choices=["CodeLlama-70b-Instruct-hf", "gemma-7b-it", "Mixtral-8x7B-Instruct-v0.1"]
                        )
    # parser.add_argument("--regenerate_path", type=str, default="results/CodeLlama-70b-Instruct-hf-apps_competition.jsonl", help="path to the jsonl file containing the samples")

    parser = parser.parse_args()

    model_name = parser.model_name
    difficulty = parser.difficulty
    model_di = MODEL_PATH[model_name]

    asyncio.run(main())
