"""
Script to generate candidate target sets. For the allowed methods, please see config.ALLOWED_METHODS_DICT; default is 'Top-K+BiMaps'
"""


import os
import argparse
from typing import Dict, Union
import torch
from sentence_transformers.util import cos_sim
from utils.helpers import load_from_pkl, save_to_pkl
from config import ALLOWED_METHODS_DICT


def sort_a2b(a: Dict, b: Dict) -> Dict:
    """
    Given the source and target dicts, for each source codes:
        a. calculates the cosine simiarliy against all target codes, 
        b. sorts the target codes in decreasing order based on the cosine similarity score.

    E.g. if src={a:xx, b:xx} and tgt = {c:xx, d:xx} then {a: [d, c], b: [c, d]} => cos_sim(a, d) > cos_sim(a, c) and cos_sim(b, c) > cos_sim(b, d);
    here xx is the code embeddings.
    """
    ## get the embeddings and stack them row-wise
    a_emb = torch.stack(list(a.values())).to('cuda')
    b_emb = torch.stack(list(b.values())).to('cuda')

    ## get the src and tgt codes
    a_keys = list(a.keys())
    
    sim_scores = cos_sim(a_emb, b_emb)
    a2b = {}
    for i in range(sim_scores.size(0)):
        score = sim_scores[i]
        score = score.sort(descending=True)
        a2b[a_keys[i]] = score

    return a2b


def gen_maps_using_topk(src: Dict, tgt: Dict, k: int) -> Dict:
    src2tgt = sort_a2b(src, tgt)
    tgt_keys = list(tgt.keys())

    maps = {}
    for key, v in src2tgt.items():
        maps[key] = [tgt_keys[idx] for idx in v.indices.tolist()[:k]]
    
    return maps


def gen_maps_using_threshold(src: Dict, tgt: Dict, th: float, num_k: Union[int, None] = None) -> Dict:
    src2tgt = sort_a2b(src, tgt)
    tgt_keys = list(tgt.keys())

    maps = {}
    for k, v in src2tgt.items():
        mask = v.values >= th
        tgt_indices = v.indices[mask].tolist()
        _maps = [tgt_keys[idx] for idx in tgt_indices]
        _maps = _maps[:num_k] if num_k else _maps
        maps[k] = _maps

    
    return maps


def gen_maps_using_bidir_map_ext(src: Dict, tgt: Dict) -> Dict:

    src2tgt = gen_maps_using_topk(src, tgt, k=1)
    tgt2src = gen_maps_using_topk(tgt, src, k=1)
    
    maps = {}
    for k, v in src2tgt.items():
        _maps = v
        for _k, _v in tgt2src.items():
            if k in _v:
                _maps.append(_k)
        
        maps[k] = list(set(_maps))
    
    return maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='Source ICD version')
    parser.add_argument('tgt', type=str, help='Target ICD version')
    parser.add_argument('ch', type=str, help='Chapter name, e.g. inf for infectious diseases')
    parser.add_argument('method', type=str, help='Method to generate maps (e.g. "th" for threshold, "oto" for using one-to-one maps or "mixed" for using both "th" and "oto")')
    parser.add_argument('--th', type=float, default=0.95, help="Threshold to include target items as potential maps. Default 0.95.")
    parser.add_argument('--k', type=int, default=3, help='Value for Top-K nearest target codes')
    args = parser.parse_args()


    if args.method not in ALLOWED_METHODS_DICT.keys():
        raise ValueError(f'Invalid method provided. Valid options are {ALLOWED_METHODS_DICT.keys()}')

    ## make output directory and out_filepath
    output_dir = os.path.join('./out/blocking', args.ch, ALLOWED_METHODS_DICT.get(args.method))
    if args.method == 'th':
        output_dir = f'{output_dir}/{str(args.th)}'
    
    if args.method in ['topk', 'default']:
        output_dir = f'{output_dir}/{str(args.k)}'

    os.makedirs(output_dir, exist_ok=True)

    ## load the corresponding source and target embeddings
    src = load_from_pkl(f'./out/emb/{args.src}_{args.ch}.pkl')
    tgt = load_from_pkl(f'./out/emb/{args.tgt}_{args.ch}.pkl')

    if args.method == 'th':
        maps = gen_maps_using_threshold(src, tgt, args.th)
    
    elif args.method == 'topk':
        maps = gen_maps_using_topk(src, tgt, args.k)

    elif args.method == 'bdm':
        maps = gen_maps_using_bidir_map_ext(src, tgt)
    
    else:
        ## default we use top-K + bdm
        if os.path.exists(f'./out/blocking/{args.ch}/Top-K/{args.k}/{args.src}_{args.tgt}.pkl'):
            maps_k = load_from_pkl(f'./out/blocking/{args.ch}/Top-K/{args.k}/{args.src}_{args.tgt}.pkl')
        else:
            maps_k = gen_maps_using_topk(src, tgt, args.k)

        if os.path.exists(f'./out/blocking/{args.ch}/Bi-Maps-Exp/{args.src}_{args.tgt}.pkl'):
            maps_oto = load_from_pkl(f'./out/blocking/{args.ch}/Bi-Maps-Exp/{args.src}_{args.tgt}.pkl')
        else:
            maps_oto = gen_maps_using_bidir_map_ext(src, tgt)
            save_to_pkl(maps_oto, f'./out/blocking/{args.ch}/Bi-Maps-Exp/{args.src}_{args.tgt}.pkl')
        maps = {}
        for k, v in maps_k.items():
            v_oto = {}
            if k in list(maps_oto.keys()):
                v_oto = maps_oto.get(k)
            
            maps[k] = list(set(v) | set(v_oto))

    save_to_pkl(data=maps, path=os.path.join(output_dir, f'{args.src}_{args.tgt}.pkl'))