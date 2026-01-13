## Balancing Precision, Recall, and Coverage in Automatic ICD Mapping
When mapping disease classification systems, such as ICD, some source code may map to multiple different target codes. This happens when a source code description is broader in meaning than  individual target code descriptions, and two or more target codes may better approximate the source code, resulting in $1:N$ mappings. Traditional embedding-based selection methods, such as threshold-based and top-$K$ selection methods, offer a potential solution to address the cardinality of the maps; however, these methods suffer from inherent trade-offs among precision, recall and mapping coverage. To this end, we propose two methods (bidirectional mapping expansion (BiMaps) and blocking-and-matching framework) to balance these trade-offs. Our experiments on mapping between different ICD versions and across different chapters suggest our proposed methods achieve high precision without sacrificing recall and achieving a broader mapping coverage (often a complete coverage in most cases).

## Usage Instructions
### 1. Getting Started
The following packages must be installed prior to running the scripts
```bash
accelerate==1.9.0
matplotlib==3.10.3
numpy==2.3.1
pandas==2.3.1
sentence-transformers==5.0.0
torch==2.7.1
transformers==4.53.2
tqdm==4.67.1
```

We have include the `Pipfile` and `Pipfile.lock` files. So, if you are managing python virtual environments using `pipenv` then you can simply use `pipevn install` to install the above dependency packages.

We ran all our experiments on a single NVIDIA A100 GPU with `cuda-12.6`.

### 2. Generating embeddings and summaries
You can generate the embeddings from using `gen_emp.py` script. The default encoder model is set to 'all-mpnet-base-v2'. To use a different Sentence-Transformer models (or hugging-face encoder models), replace the model-id with corresponding model-id in `config.ENCODER_MODEL_ID`.

Similarly, use `gen_summary.py` script to generate the LLM-generated summaries. It is important to note that, the current implementation of the `EmbeddingGenerator` class requires all the summaries, across all the ICD versions and chapters, concatinated to a single `.tsv` file. So, although it is possible to generate summaries for individual chapters, all the generated summaries must be placed in a single file. 

*(Note: For maximum reproducibility, we provide the LLM-generated summaries used in our experiments. If you choose to re-generate the descriptions, note that LLMs may produce slight variations in output even with identical prompts, which could lead to minor differences in performance metrics. We recommend using the provided summaries to replicate our exact results.)*

### 3. Generating Candidate Sets
Use the `blocking.py` script to generate the candidate sets of target codes. For the allowed methods, the current implementation supports, threshold-based (th), top-K selection (topk), bidirectional mapping expansion (bdm) and Top-K+BiMaps (default).

For example, the following script generates candidate sets using top-K ($K=10$) selection method for ICD-9-CM to ICD-10-CM (for Disease of the Digestive System chapter).
```bash
python blocking.py 'icd9cm' 'icd10cm' 'dig' 'topk' --k 10
```

### 4. LLM-based Matching
For LLM-based matching, use the `matching.py` script. The default reasoning model is set to `Qwen3-8B`; you can replace it with a different model by setting `config.SELECTOR_MODEL_ID`.

For example, the following script generates the maps using LLM reasoning for ICD-9-CM to ICD-10-CM for disease of the Digestive System chapter.
```bash
python3 matching.py 'icd9cm' 'icd10cm' 'dig' --method 'default' --k 5 --batch-size 8 --logging
```

### 5. Generating Statistics and Plots
Use `gen_statistics.py` to generate the complete statistics on the mapping performance as follows:

- Generate statistics for maps generated only using the blocking stage. `--use-macro` to calculate macro-averaged statistics.
```
python gen_statistics.py 'dig' 'default' --k 5
python gen_statistics.py 'dig' 'default' --k 5 --use-macro
```

- Generate statistics for maps generated using LLM-based matching. `--use-macro` to calculate macro-averaged statistics.
```
python gen_statistics.py 'dig' 'default' --matching --k 5
python gen_statistics.py 'dig' 'default' --matching --k 5 --use-macro
```

To generate the plot comparising the precision-recall for different block size run the `gen-plots.py` script. (Note: that the current implementation requires manual entering of the precision and recall values for each cases.)