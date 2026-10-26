"""
Script to generate final mappings using LLM-based matching. In particular, we create a multi-choice question (MCQ) prompt with the source ICD code description as the anchor term
and use items from the candidate sets (obtained from the blocking strategy), as the potential options.
We (explicitly) instruct the LLM either to select all the relavent options (and return it as a python list like string with the corresponding target code as the elements) or 
reject all the options (if none are related to the anchor term and return an empty python list like string.)
"""

import os
import argparse
from typing import List, Union, Any
from tqdm import tqdm
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.helpers import load_from_pkl, save_to_pkl, create_chat_msg, create_chat_template
from config import SELECTOR_MODEL_ID

def __gen_batched_data(d: List, batch_size: int):
    return [dict(d[i:i + batch_size]) for i in range(0, len(d), batch_size)]


def __chat_template(prompts: List[str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> List[Any]:
    msgs = [create_chat_msg(prompt) for prompt in prompts]
    return [create_chat_template(tokenizer, msg, enable_thinking=True) for msg in msgs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, required=True, help="number of current run")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size to process in a single pass. Default 8")
    parser.add_argument('--logging', action="store_true", default=False)
    args = parser.parse_args()


    data = load_from_pkl('./data/default_prompts.pkl')
    blocks = __gen_batched_data(data, args.batch_size)

    ## setup the output dir
    output_dir = f'./out/matching/prompting/{args.run}'
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SELECTOR_MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(SELECTOR_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    ## setup up logging
    if args.logging:
        os.makedirs('./out/logs/prompting', exist_ok=True)
        logging_file = f'./out/logs/prompting/run-{args.run}.log'
        logging.basicConfig(filename=logging_file, level = logging.INFO)

    matches = {}
    for block in tqdm(blocks):

        src_keys = list(block.keys())
        prompts = list(block.values())

        chat_tempates = __chat_template(prompts, tokenizer)
        
        model_inputs = tokenizer(chat_tempates, return_tensors='pt', padding=True, padding_side='left').to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs,max_new_tokens=32768)

        
        output_ids = [generated_ids[i][len(model_inputs.input_ids[i]):].tolist() for i in range(generated_ids.size(0))]

        ## log the thinking contents
        indices = []
        for output_id in output_ids:
            try:
                indices.append(len(output_id) - output_id[::-1].index(151668))
            except ValueError:
                indices.append(0)
            
        thinking_contents = [tokenizer.decode(output_ids[i][:indices[i]], skip_special_tokens=True).strip("\n") for i in range(len(output_ids))]
        responses = [tokenizer.decode(output_ids[i][indices[i]:], skip_special_tokens=True).strip("\n") for i in range(len(output_ids))]

        if args.logging:
            for i in range(len(src_keys)):
                logging.info(prompts[i])
                logging.info('-----------------')
                logging.info(thinking_contents[i])
                logging.info('-----------------')
                logging.info(responses[i])
                logging.info("========================")

        del model_inputs, generated_ids, output_ids

        for key, maps in zip(src_keys, responses):
            matches[key] = maps
        
        del src_keys, responses, key, maps, indices, thinking_contents
        torch.cuda.empty_cache()
    

    save_to_pkl(matches, f'{output_dir}/run-{args.run}.pkl')    