import openai, os

HUMANEVAL_EOS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
NON_CODE_EOS = ["<|endoftext|>", "\n```", "\n</s>", "<|endofmask|>"]
EOS = HUMANEVAL_EOS + NON_CODE_EOS

prompt = "return sum from 1 to 100"
# construct prompt
message = (
    f"Please complete the following code snippet.\n```\n{prompt.strip()}\n```"
)

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

def chatgpt_parse(ret, prompt):
    outputs = []
    for returns in ret["choices"]:
        raw_o = returns["message"]["content"]
        if "```" in raw_o:
            gen = raw_o.split("```")[1].strip()
            if gen.startswith("python"):
                gen = gen[len("python") :].strip()
            if gen.startswith(prompt.strip()):
                suf = gen.split(prompt.strip())[-1]
                suf = remove_eos(suf)
                gen = prompt.strip() + suf
            elif find_gen_func_sig(prompt) in gen:
                # same function sign is in the prompt
                sig = find_gen_func_sig(prompt)
                pre, suf = gen.split(sig)[0], gen.split(sig)[-1]
                suf = remove_eos(suf)
                gen = pre + sig + suf
            else:
                gen = f"# CANNOT PARSE CODE SNIPPET\n{gen}"
        else:
            # cannot really handle parse just dump to file and maybe process later.
            gen = f"# CANNOT PARSE\n{raw_o}"
        outputs.append(gen)
    return outputs