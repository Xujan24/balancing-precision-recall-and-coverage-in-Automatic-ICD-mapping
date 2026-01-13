"""
Class to generate summary for ICD code descriptions.
"""
import torch
import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.helpers import create_chat_msg, create_chat_template
from config import GENERATOR_MODEL_ID


class SummaryGenerator:
    def __init__(self, data_dir: str, batch_size: int = 16 ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')

        self.batches = self.__get_batches()
    

    def generate(self) -> List[Tuple]:
        data = []
        for batch in tqdm(self.batches):
            batch = self.__format_batch(batch.values.tolist())

            keys = batch.keys()
            terms = batch.values()

            del batch

            chat_inputs = self.__gen_chat_template(terms, self.tokenizer)
            model_inputs = self.tokenizer(chat_inputs, return_tensors="pt", padding=True, padding_side='left').to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )

            output_ids = [generated_ids[i][len(model_inputs.input_ids[i]):].tolist() for i in range(generated_ids.size(0))]
            response = [self.tokenizer.decode(output_id, skip_special_tokens=True).strip("\n") for output_id in output_ids]
            
            del model_inputs, generated_ids, output_ids

            data.extend(list(zip(keys, terms, response)))
            del keys, terms, response

            torch.cuda.empty_cache()
        return data


    def __get_batches(self) -> List[pd.DataFrame]:
        df = pd.read_csv(self.data_dir, dtype=str, sep=",")
        return np.array_split(df, len(df) // self.batch_size + (len(df) % self.batch_size > 0))


    def __gen_chat_template(self, terms: List[str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> List[Any]:
        prompts = [self.__gen_prompt_sum(term) for term in terms]
        msgs = [create_chat_msg(prompt) for prompt in prompts]
        return [create_chat_template(tokenizer, msg) for msg in msgs]
    

    @staticmethod
    def __gen_prompt_sum(term: str, max_words: int = 100) -> str:
        """creates prompt to generate a short clinical description for each ICD code descriptions"""
        return f"Provide a concise clinical description (max {max_words} words) of the condition '{term}'. Include (if possible) common synonyms, known causative agents, and typically affected body parts. Avoid bullet points."


    @staticmethod
    def __format_batch(x: List[List[Union[str, float]]]) -> Dict[str, str]:
        filtered_items = {}
        for item in x:
            item = list(filter(lambda x: not isinstance(x, float) or not np.isnan(x), item))
            ## some dataset have a column with root as default value
            ## we filter them out
            item = list(filter(lambda x: x != 'root', item))
            filtered_items[item[0]] = item[1]
            del item
        
        return filtered_items