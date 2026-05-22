#!/usr/bin/env python3
"""
19_evaluate_abstraction_stability.py

Evaluate repeatability of LLM-elicited input abstractions.

This script re-scores a sampled subset of topics multiple times, optionally using
light prompt paraphrases, and reports:
- exact agreement rate,
- agreement within +/-1 Likert point,
- mean score std / range,
- pairwise Spearman correlation across runs.

Outputs are written under:
  xai_analyses_results/rebuttal_abstraction_stability/
"""

import os
import re
import json
import argparse
from itertools import combinations
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lib import instruct_model

MINIMUM_SCORE = 1
MAXIMUM_SCORE = 5

SCORE_TYPE_DESCRIPTIONS = {
    'conceptually dense': "Evaluate the conceptual density of the texts in the whole web about '{topic}'. Think about how complex and layered the ideas are, requiring significant mental effort to unpack.",
    'technically complicated': "Assess the technical complexity of the texts in the whole web about '{topic}'. Consider the extent of specialized terminology or technical details.",
    'common': "Evaluate how common the texts in the whole web are about '{topic}'. Think about how frequently it appears or how widely it's understood.",
    'socially controversial': "Evaluate the level of social controversy in the texts in the whole web about '{topic}'. Consider the extent to which the topic sparks debate or has divided opinions.",
    'unambiguous': "Assess the level of clarity or unambiguity in the texts in the whole web about '{topic}'. Consider how straightforward or universally understood the topic is.",
    'positive': "Evaluate the positivity of tone in the texts in the whole web about '{topic}'. Consider how frequently the topic is associated with positive or favorable language.",
    'negative': "Assess the prevalence of negative tone in the texts in the whole web about '{topic}'. Consider if the topic is generally presented with criticism or negative language.",
    'neutral': "Evaluate the neutrality of language in the texts in the whole web about '{topic}'. Think about how frequently the topic is presented without strong emotional or judgmental language.",
    'subject to geographical variability': "Assess the geographical variability of the texts in the whole web about '{topic}'. Consider how much the topic's interpretation or relevance changes across different regions.",
    'interdisciplinary': "Evaluate the interdisciplinarity of the texts in the whole web about '{topic}'. Think about how often the topic spans multiple fields or domains (e.g., biology + computer science, philosophy + physics).",
    'subject to time variability': "Evaluate the time variability of the texts in the whole web about '{topic}'. Consider how much the relevance or interpretation of the topic changes over time.",
}

PROMPT_VARIANTS = {
    'canonical': "{desc}",
    'web_focus': "Considering aggregate web discourse rather than a single source, {desc}",
    'direct': "For the topic '{topic}', assess whether it is {score_type} in how it is typically discussed online. Keep the same meaning as this instruction: {desc}",
}

BASE_SUFFIX = """Rate your score on a scale from {min_score} (not {score_type}) to {max_score} (very {score_type}).

Expected Output Structure:
ES: Estimated Score from 1 to 5.
SE: very Short Explanation of why you give the specific score."""

SCORE_PATTERN = re.compile(r'[*#\s"\'()]*ES[*#\s"\'()]*:[*#\s"\']*(\d+)[*#\s"\']*')


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        val = float(spearmanr(x, y).statistic)
        return 0.0 if np.isnan(val) else val
    except Exception:
        return 0.0


def parse_score(output: str) -> float:
    if not isinstance(output, str):
        return float('nan')
    match = SCORE_PATTERN.search(output)
    if not match:
        return float('nan')
    return float(match.group(1))


def build_prompt(topic: str, score_type: str, variant_name: str) -> str:
    desc = SCORE_TYPE_DESCRIPTIONS[score_type].format(topic=topic)
    template = PROMPT_VARIANTS[variant_name]
    body = template.format(desc=desc, topic=topic, score_type=score_type)
    return body + "\n\n" + BASE_SUFFIX.format(
        min_score=MINIMUM_SCORE,
        max_score=MAXIMUM_SCORE,
        score_type=score_type,
    )


def load_topics(model: str, subset_size: int, random_seed: int) -> List[Tuple[str, str]]:
    df = pd.read_csv('extracted_topics.csv')
    model_col = 'Model' if 'Model' in df.columns else 'model'
    domain_col = 'Domain' if 'Domain' in df.columns else 'domain'
    topic_col = 'Topic' if 'Topic' in df.columns else 'topic'
    df = df[df[model_col] == model].copy()
    df = df.drop_duplicates(subset=[topic_col])
    if len(df) == 0:
        raise ValueError(f'No topics found for model={model!r} in extracted_topics.csv')
    if subset_size and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=random_seed)
    df = df.sort_values([domain_col, topic_col]).reset_index(drop=True)
    return list(df[[domain_col, topic_col]].itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate repeatability of LLM-elicited abstractions.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--subset_size', type=int, default=80)
    parser.add_argument('--n_repeats', type=int, default=2)
    parser.add_argument('--variants', type=str, default='canonical,web_focus')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='xai_analyses_results/rebuttal_abstraction_stability')
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    topic_rows = load_topics(args.model, args.subset_size, args.random_seed)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    invalid = [v for v in variants if v not in PROMPT_VARIANTS]
    if invalid:
        raise ValueError(f'Unknown prompt variants: {invalid}. Allowed: {sorted(PROMPT_VARIANTS)}')

    llm_options = {
        'model': args.model,
        'temperature': 0,
        'top_p': 0,
    }

    raw_rows: List[Dict[str, Any]] = []
    for repeat_id in range(args.n_repeats):
        for variant_name in variants:
            for score_type in SCORE_TYPE_DESCRIPTIONS:
                prompts = [build_prompt(topic, score_type, variant_name) for _, topic in topic_rows]
                outputs = instruct_model(prompts, **llm_options)
                for (domain, topic), output in zip(topic_rows, outputs):
                    raw_rows.append({
                        'model': args.model,
                        'domain': domain,
                        'topic': topic,
                        'score_type': score_type,
                        'repeat_id': repeat_id,
                        'variant': variant_name,
                        'score': parse_score(output),
                        'raw_output': output,
                    })

    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(args.output_dir, f'raw_scores_{args.model}.csv')
    raw_df.to_csv(raw_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for score_type, gdf in raw_df.groupby('score_type'):
        pivot = gdf.pivot_table(
            index='topic',
            columns=['variant', 'repeat_id'],
            values='score',
            aggfunc='first',
        )
        values = pivot.to_numpy(dtype=float)
        exact_all = np.nanmean([len(set(row[~np.isnan(row)].astype(int).tolist())) == 1 for row in values if np.sum(~np.isnan(row)) > 1]) if len(values) else float('nan')
        within_one_all = np.nanmean([(np.nanmax(row) - np.nanmin(row)) <= 1 for row in values if np.sum(~np.isnan(row)) > 1]) if len(values) else float('nan')
        summary_rows.append({
            'model': args.model,
            'score_type': score_type,
            'n_topics': int(len(pivot)),
            'n_measurements_per_topic': int(pivot.shape[1]),
            'mean_score_std': float(np.nanmean(np.nanstd(values, axis=1))),
            'median_score_std': float(np.nanmedian(np.nanstd(values, axis=1))),
            'mean_score_range': float(np.nanmean(np.nanmax(values, axis=1) - np.nanmin(values, axis=1))),
            'exact_consensus_rate': float(exact_all),
            'within_one_consensus_rate': float(within_one_all),
        })

        col_names = list(pivot.columns)
        for c1, c2 in combinations(col_names, 2):
            s1 = pivot[c1].to_numpy(dtype=float)
            s2 = pivot[c2].to_numpy(dtype=float)
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if np.sum(valid) == 0:
                continue
            exact = float(np.mean(s1[valid] == s2[valid]))
            within1 = float(np.mean(np.abs(s1[valid] - s2[valid]) <= 1))
            pair_rows.append({
                'model': args.model,
                'score_type': score_type,
                'variant_a': c1[0],
                'repeat_a': int(c1[1]),
                'variant_b': c2[0],
                'repeat_b': int(c2[1]),
                'n_topics': int(np.sum(valid)),
                'exact_agreement': exact,
                'within_one_agreement': within1,
                'spearman': safe_spearman(s1[valid], s2[valid]),
                'mean_abs_diff': float(np.mean(np.abs(s1[valid] - s2[valid]))),
            })

    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)
    summary_path = os.path.join(args.output_dir, f'summary_{args.model}.csv')
    pair_path = os.path.join(args.output_dir, f'pairwise_{args.model}.csv')
    summary_df.to_csv(summary_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    final_summary = {
        'model': args.model,
        'subset_size': int(len(topic_rows)),
        'n_repeats': int(args.n_repeats),
        'variants': variants,
        'raw_path': raw_path,
        'summary_path': summary_path,
        'pairwise_path': pair_path,
        'aggregate_exact_consensus_rate': float(summary_df['exact_consensus_rate'].mean()) if len(summary_df) else float('nan'),
        'aggregate_within_one_consensus_rate': float(summary_df['within_one_consensus_rate'].mean()) if len(summary_df) else float('nan'),
        'aggregate_pairwise_spearman': float(pair_df['spearman'].mean()) if len(pair_df) else float('nan'),
    }
    json_path = os.path.join(args.output_dir, f'final_summary_{args.model}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)

    print(json.dumps(final_summary, indent=2))


if __name__ == '__main__':
    main()
