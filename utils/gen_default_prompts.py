import pandas as pd
from utils.helpers import save_to_pkl, load_from_pkl
from config import MAPPING_PAIRS, ICD_VERSIONS, CHAPTERS


def gen_prompt(src, tgt, ch, code)-> str:

    prompt = f"""
    You are a medical coding expert. Your task is to find all the equivalent codes in {ICD_VERSIONS.get(tgt)} for the given code in {ICD_VERSIONS.get(src)}. 
    The code is from the chapter "{CHAPTERS.get(ch)}".

    Source code: {code}

    Instructions:
    Return a **single Python-style list of strings**.
    - The list should contain only the {ICD_VERSIONS.get(tgt)} code not the code description.
    - The list should contain codes from equivalent target chapter in {ICD_VERSIONS.get(tgt)}; if no equivalent code is found, return an empty list.
    - **DO NOT include any explanation, additional text, or special characters outside the list.**
    """

    return prompt.strip()


def main() -> None:
    data = []
    for pair in MAPPING_PAIRS:
        src, tgt = pair['src'], pair['tgt']
        gt = load_from_pkl(f'./data/gt/{src}_{tgt}.pkl')
        
        for ch in list(CHAPTERS.keys()):
            src_ls = pd.read_csv(f'./data/icd/{src}_{ch}.csv', dtype=str).values.tolist()
            tgt_ls = pd.read_csv(f'./data/icd/{tgt}_{ch}.csv', dtype=str).values.tolist()

            src_ls = [[item for item in row if pd.notna(item)] for row in src_ls]
            tgt_ls = [[item for item in row if pd.notna(item)] for row in tgt_ls]

            tgt_codes = [item[0] for item in tgt_ls]

            for item in src_ls:
                key = item[0]
                code = f'{item[1]} ({item[0]})'

                ## will skip source codes that do not have any valid mappings in the ground truth mappings or
                ## maps completely to a different chapter in the target system.
                if key not in list(gt.keys()) or len(set(gt.get(key)).intersection(set(list(tgt_codes)))) == 0:
                    continue
                
                data.append((f"{key}#{src}-{tgt}-{ch}", gen_prompt(src, tgt, ch, code)))

    save_to_pkl(data, './data/default_prompts.pkl')


if __name__ == "__main__":
    main()
