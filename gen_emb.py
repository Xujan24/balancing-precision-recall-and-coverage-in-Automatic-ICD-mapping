"""
Script to generate embeddings
"""

import os
import argparse
from utils.emb_generator import EmbeddingGenerator
from utils.helpers import save_to_pkl
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ## setup the output path
    output_dir = f'./out/emb'
    os.makedirs(output_dir, exist_ok=True)

    filenames = os.listdir('./data/icd')

    for filename in filenames:
        basename = filename.split('.')[0]
        tqdm.write(f"Generating embeddings for {basename}")

        emb_generator = EmbeddingGenerator(f'./data/icd/{filename}', args.batch_size)
        embs = emb_generator.gen_embeddings()

        save_to_pkl(embs, os.path.join(output_dir, f'{basename}.pkl'))