from typing import List, Union, Any, Dict
import ast
import pickle as pkl
import numpy as np
from transformers import  PreTrainedTokenizer, PreTrainedTokenizerFast


def load_from_pkl(path: str) -> Any:
    with open(path, 'rb') as f:
        data = pkl.load(f)

    return data


def save_to_pkl(data: any, path: str) -> None:
    with open(path, 'wb') as f:
        pkl.dump(obj=data, file=f)
    
    return None


def parse_string_as_list(s):
    """
    Function to parse the list-like string to a list.
    """
    try:
        # First try literal_eval for properly quoted lists
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass

    # Fallback: remove brackets, split by whitespace
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if inner:
            return inner.split()
        else:
            return []
    else:
        return [s]


def filter_nan(row: List[List[Union[str, float]]]) -> List[List[str]]:
    """Filter Nan values from a list"""
    filtered_items = []
    for item in row:
        item = list(filter(lambda x: not isinstance(x, float) or not np.isnan(x), item))
        ## in our dataset some have a column with root as default value
        ## we filter them out
        item = list(filter(lambda x: x != 'root', item))
        filtered_items.append(item)
        del item
    
    return filtered_items



def create_chat_msg(prompt: str) -> List[Dict[str, str]]:
    """Formats the prompt in chat format"""
    return [{"role": "user", "content": prompt}]


def create_chat_template(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], msg: List[Dict[str, str]], enable_thinking: bool = False) -> Any:
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)