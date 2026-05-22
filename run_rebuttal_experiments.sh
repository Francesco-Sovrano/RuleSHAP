#!/usr/bin/env bash
set -euo pipefail

# Activate local environment if present.
if [ -f .env/bin/activate ]; then
  . .env/bin/activate
fi

MODEL="${1:-llama3.1}"

# 1) Held-out validation of naturally occurring rules in the no-injection setting.
echo 16_evaluate_no_injection_generalization
python 16_evaluate_no_injection_generalization.py \
  --model "$MODEL" \
  --difficulty baseline \
  --methods ruleshap,rulefit \
  --metrics explanation_length,subjectivity_score_nn,gunning_fog,sentiment_score_nn \
  --n_splits 5 \
  --test_size 0.2 \
  --top_k_rules 5

# 2) Robustness of nearest-neighbor SHAP imputation.
echo 17_evaluate_shap_imputation_robustness
python 17_evaluate_shap_imputation_robustness.py \
  --model "$MODEL" \
  --difficulty "easy" \
  --n_samples 256 \
  --n_masks_per_sample 32 

# 2) Robustness of nearest-neighbor SHAP imputation for medium difficulty.
echo 17_evaluate_shap_imputation_robustness
python 17_evaluate_shap_imputation_robustness.py \
  --model "$MODEL" \
  --difficulty "medium" \
  --n_samples 256 \
  --n_masks_per_sample 32 

# 2) Robustness of nearest-neighbor SHAP imputation for hard difficulty.
echo 17_evaluate_shap_imputation_robustness
python 17_evaluate_shap_imputation_robustness.py \
  --model "$MODEL" \
  --difficulty "hard" \
  --n_samples 256 \
  --n_masks_per_sample 32 

# 3) GELPE baseline extraction and evaluation.
echo 18_evaluate_gelpe
######################################################
### GELPE baseline extraction

### GPT-3.5
python 7_extract_rules_with_gelpe.py --model gpt-3.5-turbo --difficulty baseline &
python 7_extract_rules_with_gelpe.py --model gpt-3.5-turbo --difficulty easy &
python 7_extract_rules_with_gelpe.py --model gpt-3.5-turbo --difficulty medium &
python 7_extract_rules_with_gelpe.py --model gpt-3.5-turbo --difficulty hard

######################################################

######################################################
### GPT-4o-mini
python 7_extract_rules_with_gelpe.py --model gpt-4o-mini --difficulty baseline &
python 7_extract_rules_with_gelpe.py --model gpt-4o-mini --difficulty easy &
python 7_extract_rules_with_gelpe.py --model gpt-4o-mini --difficulty medium &
python 7_extract_rules_with_gelpe.py --model gpt-4o-mini --difficulty hard

######################################################

######################################################
### GPT-4o
python 7_extract_rules_with_gelpe.py --model gpt-4o --difficulty baseline &
python 7_extract_rules_with_gelpe.py --model gpt-4o --difficulty easy &
python 7_extract_rules_with_gelpe.py --model gpt-4o --difficulty medium &
python 7_extract_rules_with_gelpe.py --model gpt-4o --difficulty hard

######################################################

######################################################
### llama3.1
python 7_extract_rules_with_gelpe.py --model llama3.1 --difficulty baseline &
python 7_extract_rules_with_gelpe.py --model llama3.1 --difficulty easy &
python 7_extract_rules_with_gelpe.py --model llama3.1 --difficulty medium &
python 7_extract_rules_with_gelpe.py --model llama3.1 --difficulty hard

######################################################

######################################################
### llama3.1:70b
python 7_extract_rules_with_gelpe.py --model llama3.1:70b --difficulty baseline &
python 7_extract_rules_with_gelpe.py --model llama3.1:70b --difficulty easy &
python 7_extract_rules_with_gelpe.py --model llama3.1:70b --difficulty medium &
python 7_extract_rules_with_gelpe.py --model llama3.1:70b --difficulty hard

######################################################

######################################################
### Evaluate GELPE
python 18_evaluate_gelpe.py

# 4) Repeatability of abstraction scoring under repeated/paraphrased prompts.
echo 19_evaluate_abstraction_stability
python 19_evaluate_abstraction_stability.py \
  --model "$MODEL" \
  --subset_size 80 \
  --n_repeats 2 \
  --variants canonical,web_focus

# 5) Sensitivity of semantic duplicate removal to encoder / threshold choices.
echo 20_evaluate_dedup_sensitivity
python 20_evaluate_dedup_sensitivity.py \
  --models "$MODEL" \
  --encoders all-MiniLM-L6-v2,all-mpnet-base-v2 \
  --thresholds 0.85,0.90,0.95 \
  --group_by model
