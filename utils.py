import openai
import re
import six
import spacy
from nltk.stem.porter import PorterStemmer
import ast
from zss import Node
from zss import simple_distance

PorterStemmer = PorterStemmer()
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

def get_openai_response(prompt: str, max_tokens = 150, temperature = 0.7, top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature = temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    output = response['choices'][0]['text']
    assert output.startswith(prompt)
    gen_text = output[len(prompt):].strip()
    return gen_text

def get_davinci003_response(prompt_text: str, max_tokens = 150, temperature = 0.7, top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt_text,
                                        max_tokens=max_tokens,
                                        temperature = temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    # output = response['choices'][0]['text']
    # assert output.startswith(prompt)
    # gen_text = output[len(prompt):].strip()
    return response

def get_chatgpt_qa_response(prompt_text, temperature = 0.7, max_tokens=1000):
    messages = [{"role":"system", "content": "You are a helpful assistant that answers the question provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_gpt4_qa_response(prompt_text, temperature = 0.7, max_tokens=1000):
    messages = [{"role":"system", "content": "You are a helpful assistant that answers the question provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-4-0314",
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_gpt4_completion_response(prompt_text, max_tokens):
    messages = [{"role":"system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-4-0314",
                messages = messages,
                temperature = 0.7,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_chatgpt_completion_response(prompt_text, max_tokens):
    messages = [{"role":"system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                temperature = 0.7,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def tokenize(text, stemmer, stopwords=[]):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

    Returns:
    A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

    tokens = re.split(r"\s+", text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens if x not in stopwords]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

    return tokens


HUMANEVAL_EOS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
NON_CODE_EOS = ["<|endoftext|>", "\n```", "\n</s>", "<|endofmask|>"]
EOS = HUMANEVAL_EOS + NON_CODE_EOS

def find_gen_func_sig(prompt):
    func_sig = ""
    for x in prompt.splitlines():
        if x.startswith("def ") and x.endswith(":"):
            # always pick the last one, since there could pre-defined functions.
            func_sig = x
    return func_sig

def remove_eos(gen):
    min_index = 1000
    for eos in EOS:
        if eos in gen:
            min_index = min(min_index, gen.index(eos))
    return gen[:min_index]

def parse_code_snippet(prompt, raw_o):
    if "```" in raw_o:
        gen = raw_o.split("```")[1].strip()
        if gen.startswith("python"):
            gen = gen[len("python") :].strip()
        if gen.startswith(prompt.strip()):
            suf = gen.split(prompt.strip())[-1]
            suf = remove_eos(suf)
            gen = prompt.strip() + suf
        elif find_gen_func_sig(prompt) == '':
            gen = gen
        elif find_gen_func_sig(prompt) in gen:
            # same function sign is in the prompt
            sig = find_gen_func_sig(prompt)
            pre, suf = gen.split(sig)[0], gen.split(sig)[-1]
            suf = remove_eos(suf)
            gen = pre + sig + suf
        else:
            gen = f"# CANNOT PARSE CODE SNIPPET\n{gen}"
            print('CANNOT PARSE CODE SNIPPET')
    elif raw_o.startswith('def') or raw_o.startswith('import') or raw_o.startswith('from'):
        gen = raw_o
    else:
        # cannot really handle parse just dump to file and maybe process later.
        gen = f"# CANNOT PARSE\n{raw_o}"
        print('CANNOT PARSE CODE SNIPPET')
    return gen




class ASTToZSSConverter(ast.NodeVisitor):
    def visit(self, node):
        new_node = Node(label=str(type(node).__name__))
        for child in ast.iter_child_nodes(node):
            new_node.addkid(self.visit(child))

        return new_node

converter = ASTToZSSConverter()

def tree_original_distance( code1 ):
    code_none = ''''''
    zss_tree0 = converter.visit(ast.parse(code_none))
    zss_tree1 = converter.visit(ast.parse(code1))
    
    distance = simple_distance(zss_tree1, zss_tree0)
    return distance

def tree_edit_distance( code1, code2 ):
    zss_tree1 = converter.visit(ast.parse(code1))
    zss_tree2 = converter.visit(ast.parse(code2))
    
    distance = simple_distance(zss_tree1, zss_tree2)
    return distance

