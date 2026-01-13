"""
Script to generate final mappings using LLM-based matching. In particular, we create a multi-choice question (MCQ) prompt with the source ICD code description as the anchor term
and use items from the candidate sets (obtained from the blocking strategy), as the potential options.
We (explicitly) instruct the LLM either to select all the relavent options (and return it as a python list like string with the corresponding target code as the elements) or 
reject all the options (if none are related to the anchor term and return an empty python list like string.)
"""

import os
import argparse
import pandas as pd
from typing import Dict, List, Union, Any
from tqdm import tqdm
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.helpers import load_from_pkl, save_to_pkl, create_chat_msg, create_chat_template, filter_nan
from config import ALLOWED_METHODS_DICT, SELECTOR_MODEL_ID, NEW_LINE


def __get_code_desc_dict(df: pd.DataFrame) -> Dict:
    return {a[0]: a[1] for a in filter_nan(df.values.tolist())}


def __gen_block_batches(blocks: Dict, batch_size: int) -> List[Dict]:
    items = list(blocks.items())
    batches = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        batches.append(dict(batch_items))
        del batch_items
    return batches


def __gen_prompt(src_desc: str, options: List) -> str:
    prompt = f"""
    You are a medical coding expert tasked with identifying semantically equivalent codes.
    Source Code Description:
    {src_desc}
    Select ALL options that represent the same or highly similar clinical condition as the source.
    {NEW_LINE.join(options)}
    Instructions:
    Return a **single Python-style list of strings**.
    - The list should contain only the matching `code` values.
    - If none of the options match, return `[]`.
    - **DO NOT include any explanation, additional text, or special characters outside the list.**
    """

    return prompt.strip()


def __chat_template(prompts: List[str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> List[Any]:
    msgs = [create_chat_msg(prompt) for prompt in prompts]
    return [create_chat_template(tokenizer, msg, enable_thinking=True) for msg in msgs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='Source ICD version e.g. icd9cm')
    parser.add_argument('tgt', type=str, help='Target ICD version e.g. icd10cm')
    parser.add_argument('ch', type=str, help='Chapter index ["dig", "inf", "resp"]')
    parser.add_argument('--method', type=str, default='default', help='Methods used to create the blocks. Default \'default\'')
    parser.add_argument('--k', type=int, default=5, help='Number of top target codes to select to generate the blocks; Default 5. Discard if using \'th\' or \'oto+th\'')
    parser.add_argument('--th', type=float, default=0.65, help='Cutoff or the threshold value for the cosine similarity scores; Default .85. Discard if using \'topk\' or \'oto+topk\'')
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size to process in a single pass. Default 8")
    parser.add_argument('--logging', action="store_true", default=False)
    args = parser.parse_args()

    if args.method not in list(ALLOWED_METHODS_DICT.keys()):
        raise ValueError(f'Invalid method provided. Allowed methods are: {list(ALLOWED_METHODS_DICT.keys())}')
    
    
    src_df = pd.read_csv(f'./data/icd/{args.src}_{args.ch}.csv', dtype=str)
    tgt_df = pd.read_csv(f'./data/icd/{args.tgt}_{args.ch}.csv', dtype=str)

    src_dict = __get_code_desc_dict(src_df)
    tgt_dict = __get_code_desc_dict(tgt_df)

    del src_df, tgt_df

    ## load the blocks
    blocks_dir = f'./out/blocking/{args.ch}/{ALLOWED_METHODS_DICT.get(args.method)}'
    if args.method == 'th':
        blocks_dir = f'{blocks_dir}/{args.th}' 
    
    if args.method in ['topk', 'default']:
        blocks_dir = f'{blocks_dir}/{args.k}'
    
    blocks = load_from_pkl(f'{blocks_dir}/{args.src}_{args.tgt}.pkl')

    ## to minimize the stress on compute, lets filter out those source codes for which there are no mappings in the ground truth
    ## also lets discard those cases where a source code is mapped entirely to a target code in a different chapter.
    ## we will discard these cases anyways when calculating the statistics
    ## however, this could be interesting for analyzing, which we leave for the future
    gt_dict = load_from_pkl(f'./data/gt/{args.src}_{args.tgt}.pkl')

    filtered_blocks = {}
    for k, v in blocks.items():
        if k not in list(gt_dict.keys()):
            ## no mapping exists in the ground truth
            continue
        
        if len(set(gt_dict.get(k)).intersection(set(list(tgt_dict.keys())))) == 0:
            ## maps to different chapter
            continue

        filtered_blocks[k] = v
    
    del blocks, blocks_dir, gt_dict

    blocks = __gen_block_batches(filtered_blocks, args.batch_size)

    ## setup the output dir
    output_dir = f'./out/matching/{args.ch}/{ALLOWED_METHODS_DICT.get(args.method)}'
    if args.method == 'th':
        output_dir = f'{output_dir}/{args.th}'

    if args.method in ['topk', 'default']:
        output_dir = f'{output_dir}/{args.k}'

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SELECTOR_MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(SELECTOR_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    ## setup up logging
    if args.logging:
        os.makedirs('./out/logs', exist_ok=True)
        logging_file = f'./out/logs/{args.method}_{args.src}_{args.tgt}_{args.ch}_{args.k}.log'
        logging.basicConfig(filename=logging_file, level = logging.INFO)

    matches = {}
    for block in tqdm(blocks):
        src_keys = list(block.keys())
        tgt_keys = list(block.values())

        ## filter out one-to-one mapping cases (in the predicted maps)
        idx_1to1 = [len(tgt_key) <= 1 for tgt_key in tgt_keys]
        oto_src_keys = [src_key for src_key, mask in zip(src_keys, idx_1to1) if mask]
        oto_tgt_keys = [tgt_key for tgt_key, mask in zip(tgt_keys, idx_1to1) if mask]

        ## add all one-to-one mapping into the matches dict.
        for _k, _m in zip(oto_src_keys, oto_tgt_keys):
            matches[_k] = _m

        src_keys = [src_key for src_key, mask in zip(src_keys, idx_1to1) if not mask]
        tgt_keys = [tgt_key for tgt_key, mask in zip(tgt_keys, idx_1to1) if not mask]

        if len(src_keys) == 0 or len(tgt_keys) == 0:
            continue

        src_keys_desc = [src_dict.get(k) for k in src_keys]
        tgt_keys_desc = [[f'{a} {tgt_dict.get(a)}' for a in k] for k in tgt_keys]
        
        prompts = [__gen_prompt(anchor, options) for anchor, options in zip(src_keys_desc, tgt_keys_desc)]
        del src_keys_desc, tgt_keys_desc, tgt_keys

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
    

    save_to_pkl(matches, f'{output_dir}/{args.src}_{args.tgt}.pkl')    