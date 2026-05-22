#!/usr/bin/env python3
"""
20_evaluate_dedup_sensitivity.py

Evaluate how sensitive the already-extracted topic pool is to the choice of
semantic encoder and duplicate-removal threshold.

Given the current extracted topic set, this script re-applies semantic de-duplication
under alternative settings and reports retained counts, retention ratios, and pairwise
Jaccard overlap between settings.

Outputs are written under:
  xai_analyses_results/rebuttal_dedup_sensitivity/
"""

import os
import json
import argparse
from itertools import combinations
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


def normalize_topics(topics: List[str]) -> List[str]:
    return sorted({t.strip() for t in topics if isinstance(t, str) and t.strip()}, key=lambda x: (len(x), x.lower()))


def dedup_topics(topics: List[str], encoder_name: str, threshold: float) -> List[str]:
    topics = normalize_topics(topics)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError('sentence_transformers is required for 20_evaluate_dedup_sensitivity.py. Install requirements.txt first.') from e
    model = SentenceTransformer(encoder_name)
    emb = np.asarray(model.encode(topics, convert_to_tensor=False, show_progress_bar=False), dtype=np.float32)
    keep_indices: List[int] = []
    for i in range(len(topics)):
        keep = True
        for j in keep_indices:
            a = emb[i]
            b = emb[j]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if sim > threshold:
                keep = False
                break
        if keep:
            keep_indices.append(i)
    return [topics[i] for i in keep_indices]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    parser = argparse.ArgumentParser(description='Re-apply de-duplication with alternative encoders/thresholds.')
    parser.add_argument('--models', type=str, default='')
    parser.add_argument('--domains', type=str, default='')
    parser.add_argument('--encoders', type=str, default='all-MiniLM-L6-v2,all-mpnet-base-v2')
    parser.add_argument('--thresholds', type=str, default='0.85,0.90,0.95')
    parser.add_argument('--group_by', choices=['model', 'domain', 'model_domain', 'all'], default='model')
    parser.add_argument('--output_dir', type=str, default='xai_analyses_results/rebuttal_dedup_sensitivity')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv('extracted_topics.csv')
    model_col = 'Model' if 'Model' in df.columns else 'model'
    domain_col = 'Domain' if 'Domain' in df.columns else 'domain'
    topic_col = 'Topic' if 'Topic' in df.columns else 'topic'

    if args.models.strip():
        model_filter = {x.strip() for x in args.models.split(',') if x.strip()}
        df = df[df[model_col].isin(model_filter)]
    if args.domains.strip():
        domain_filter = {x.strip() for x in args.domains.split(',') if x.strip()}
        df = df[df[domain_col].isin(domain_filter)]
    if len(df) == 0:
        raise ValueError('No topics left after applying model/domain filters.')

    if args.group_by == 'model':
        groups = [(str(model), subdf) for model, subdf in df.groupby(model_col)]
    elif args.group_by == 'domain':
        groups = [(str(domain), subdf) for domain, subdf in df.groupby(domain_col)]
    elif args.group_by == 'model_domain':
        groups = [(f'{model}__{domain}', subdf) for (model, domain), subdf in df.groupby([model_col, domain_col])]
    else:
        groups = [('all_topics', df)]

    encoders = [x.strip() for x in args.encoders.split(',') if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]

    counts_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []

    for group_name, subdf in groups:
        topics = normalize_topics(subdf[topic_col].tolist())
        setting_topics: Dict[Tuple[str, float], List[str]] = {}
        for encoder_name in encoders:
            for threshold in thresholds:
                retained = dedup_topics(topics, encoder_name=encoder_name, threshold=threshold)
                setting_topics[(encoder_name, threshold)] = retained
                counts_rows.append({
                    'group': group_name,
                    'encoder': encoder_name,
                    'threshold': threshold,
                    'n_original_topics': int(len(topics)),
                    'n_retained_topics': int(len(retained)),
                    'retention_ratio': float(len(retained) / len(topics)) if topics else 0.0,
                })
                safe_encoder = encoder_name.replace('/', '_')
                out_path = os.path.join(args.output_dir, f'topics_{group_name}_{safe_encoder}_thr{threshold:.2f}.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(retained, f, indent=2)
        for (enc_a, thr_a), (enc_b, thr_b) in combinations(setting_topics.keys(), 2):
            overlap_rows.append({
                'group': group_name,
                'encoder_a': enc_a,
                'threshold_a': thr_a,
                'encoder_b': enc_b,
                'threshold_b': thr_b,
                'jaccard': float(jaccard(setting_topics[(enc_a, thr_a)], setting_topics[(enc_b, thr_b)])),
                'count_delta': int(len(setting_topics[(enc_a, thr_a)]) - len(setting_topics[(enc_b, thr_b)])),
            })

    counts_df = pd.DataFrame(counts_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    counts_path = os.path.join(args.output_dir, 'counts.csv')
    overlap_path = os.path.join(args.output_dir, 'overlap.csv')
    counts_df.to_csv(counts_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    summary = {
        'group_by': args.group_by,
        'encoders': encoders,
        'thresholds': thresholds,
        'counts_path': counts_path,
        'overlap_path': overlap_path,
        'mean_retention_ratio': float(counts_df['retention_ratio'].mean()) if len(counts_df) else float('nan'),
        'mean_pairwise_jaccard': float(overlap_df['jaccard'].mean()) if len(overlap_df) else float('nan'),
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
