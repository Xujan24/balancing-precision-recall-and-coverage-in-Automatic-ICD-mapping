"""
Script to generate the summaries.

Note: 
    It is best to first create a single .csv file by combining all chapter (and ICD version) files with two columns ['codes', 'code_desc'].
    This way all the generated summaries will be in a single file. However, if you generated the summaries individually, then you must
    combine them all to a single file. This is because the current implementation for EmbeddingGenerator requires the summaries in a single file.
"""
import os
import argparse
import pandas as pd
from utils.summary_generator import SummaryGenerator


OUT_FILENAME = 'summary.tsv'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='path to input .csv file')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    output_dir = f'./out/summary'
    os.makedirs(output_dir, exist_ok=True)

    generator = SummaryGenerator(args.src, args.batch_size)
    summaries = generator.generate()

    data_df = pd.DataFrame(summaries, index=None, columns=['code', 'code_desc', 'summary'])
    data_df.to_csv(os.path.join(output_dir, OUT_FILENAME), index=None, sep="\t")