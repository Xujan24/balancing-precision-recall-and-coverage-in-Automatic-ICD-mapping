"""
Class to generate the ICD code embeddings using hierarchy-augmented (HA) and LLM-generated (LG) descriptions/summaries.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict
from utils.helpers import filter_nan
from config import ENCODER_MODEL_ID

class EmbeddingGenerator:
    def __init__(self, src: str, batch_size: int = 16) -> None:
        self.src = src
        self.batch_size = batch_size
        
        self.encoder = SentenceTransformer(ENCODER_MODEL_ID)

        self.batches = self.__get_batches()
        self.summaries = self.__get_summaries()


    def gen_embeddings(self) -> Dict:
        embs = {}
        for batch in tqdm(self.batches):
            batch = batch.values.tolist()
            batch = filter_nan(batch)

            lg = self.__get_batch_summaries(batch, self.summaries)
            ha = self.__gen_ha_summary(batch)

            del batch

            keys = list(lg.keys())

            ## we combine both lg and ha to process in a single pass using the SBERT
            inputs = [*list(lg.values()), *list(ha.values())]

            del lg, ha

            with torch.no_grad():
                batch_embs = self.encoder.encode(inputs, convert_to_tensor=True, normalize_embeddings=True)

            emb1, emb2 = torch.split(batch_embs, split_size_or_sections=batch_embs.size(0)//2, dim=0)
            del batch_embs
            
            final_emb = self.__normalize_row((emb1 + emb2)/2)
            del emb1, emb2

            for i in range(len(keys)):
                embs[str(keys[i])] = final_emb[i].cpu()

            del keys, final_emb
            torch.cuda.empty_cache()
        
        return embs
    

    def __get_batches(self) -> List[pd.DataFrame]:
        df = pd.read_csv(self.src, dtype=str)
        return np.array_split(df, len(df) // self.batch_size + (len(df) % self.batch_size > 0))

    
    @staticmethod
    def __get_summaries() -> Dict:
        summaries = {}
        summ_df = pd.read_csv(f'./out/summary/summary.tsv', dtype=str, sep='\t')
        for row in summ_df.values.tolist():
            summaries[row[0]] = row[-1]
        
        return summaries
    

    @staticmethod
    def __normalize_row(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(p=2, dim=1, keepdim=True)


    @staticmethod
    def __gen_ha_summary(batch: List[List[str]]) -> Dict[str, str]:
        """
        Returns the hierarchical augmented (HA) descriptions.
        """
        
        prompts = {}
        for item in batch:
            key = item[0]
            item = item[1:]
            fact = ''.join([f"{item[i].lower()} is {item[i+1].lower()}. " for i in range(len(item[1:]))])
            prompts[key] = fact
            del key, item, fact
        
        return prompts


    @staticmethod
    def __get_batch_summaries(batch: List[List[str]], summaries: Dict[str, str]) -> Dict[str, str]:
        prompts = {}
        for row in batch:
            code = row[0]
            prompts[code] = summaries[code]
        return prompts