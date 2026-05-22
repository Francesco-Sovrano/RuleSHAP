# RuleSHAP: Global Rule Extraction for Auditing Injected LLM Behaviours

Official code and data for the paper **"Can Global XAI Methods Reveal Injected Behaviours in LLMs? SHAP vs Rule Extraction vs RuleSHAP"**.

The project evaluates whether global explainability methods can detect deliberately injected misinformation-related behaviours in large language model outputs. It converts text prompts and model responses into ordinal features, computes SHAP explanations, extracts global rules, and compares RuleSHAP against RuleFit, decision trees, linear models, SHAP-only rankings, and GELPE.

## Contents

- [Paper](#paper)
- [Project overview](#project-overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [External APIs and services](#external-apis-and-services)
- [Installation](#installation)
- [Model setup](#model-setup)
- [Run the experiments](#run-the-experiments)
- [Optional and rebuttal experiments](#optional-and-rebuttal-experiments)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)


## Paper

Francesco Sovrano. 2026. **Can Global XAI Methods Reveal Injected Behaviours in LLMs? SHAP vs Rule Extraction vs RuleSHAP.** In *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (KDD '26), ACM, Jeju, Republic of Korea, 12 pages. DOI: `10.1145/3770855.3818093`

## Project overview

Generative AI systems can spread useful information, but they can also amplify misleading or misinformative behaviours. Standard global XAI methods are typically designed for structured numeric data, not raw LLM input and output text. This repository addresses that gap with a text-to-ordinal workflow:

1. Build a topic set around Sustainable Development Goal domains.
2. Score each topic along interpretable ordinal dimensions, such as commonality, positivity, controversy, and interdisciplinarity.
3. Generate LLM explanations under controlled behaviour-injection instructions.
4. Convert responses into output metrics, such as explanation length, subjectivity, sentiment, and readability.
5. Compute SHAP values over the ordinal feature space.
6. Extract global rules with RuleSHAP and compare them against baseline explainability methods.

RuleSHAP combines SHAP-guided feature attribution with rule extraction so that non-univariate injected behaviours can be expressed as actionable rules.

## Repository layout

```text
.
├── 1_extract_topics.py
├── 2_get_internal_scores.py
├── 3_get_explanations.py
├── 4_get_output_metrics.py
├── 5_compute_shap_values.py
├── 6_extract_rules.py
├── 7_extract_rules_with_baselines.py
├── 7_extract_rules_with_gelpe.py
├── 8_evaluate_ruleshap.py
├── 9_evaluate_rulefit.py
├── 10_evaluate_dtree.py
├── 11_evaluate_shap.py
├── 12_input_output_correlation_analysis.py
├── 13_llm_estimate_proxy_metrics_correlation_analysis.py
├── 14_statistically_test_ruleshap_improvements_over_rulefit.py
├── 15_rule_guided_mitigation.py
├── 16_evaluate_no_injection_generalization.py
├── 17_evaluate_shap_imputation_robustness.py
├── 18_evaluate_gelpe.py
├── 19_evaluate_abstraction_stability.py
├── 20_evaluate_dedup_sensitivity.py
├── gelpe.py
├── lib.py
├── ruleshap.py
├── xai_eval_utils.py
├── requirements.txt
├── setup.sh
├── run_all_experiments.sh
└── run_rebuttal_experiments.sh
```

Key modules:

- `ruleshap.py`: RuleSHAP implementation and SHAP-weighted linear model utilities.
- `gelpe.py`: GELPE baseline utilities.
- `lib.py`: shared caching and LLM-call helpers.
- `xai_eval_utils.py`: evaluation helpers for reciprocal rank, rule matching, and fidelity summaries.

## Requirements

The project was tested with:

- macOS 15.3
- Python 3.9

Core Python packages are listed in `requirements.txt`. The experiment pipeline also requires:

- an OpenAI API key when running GPT-based models
- Ollama when running local Llama models
- spaCy and NLTK data installed by `setup.sh`

The full experiment suite is compute intensive. Running every model and difficulty level will create many CSV, pickle, and figure files.


## External APIs and services

The code can run with local models only, but several scripts support external or local model APIs through `lib.py`:

- OpenAI API: GPT-family model names use `OPENAI_API_KEY` and the `https://api.openai.com/v1` endpoint.
- Groq API: the Groq-hosted model names in `lib.py` use `GROQ_API_KEY` and the OpenAI-compatible `https://api.groq.com/openai/v1` endpoint.
- Ollama: `llama3.1` and `llama3.1:70b` use the local Ollama Python client; other non-GPT fallback models use the local OpenAI-compatible Ollama endpoint at `http://localhost:11434/v1`.
- Hugging Face model downloads: `4_get_output_metrics.py` and `15_rule_guided_mitigation.py` load Transformer pipelines for multilingual sentiment and subjectivity scoring.
- spaCy and NLTK downloads: `setup.sh` installs language resources used by the text-processing pipeline.

Generated LLM responses are cached under `cache/` when the scripts run, which reduces repeated API calls for the same prompts and settings.

## Installation

Create the environment and install dependencies with the provided setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script creates a local `.env` virtual environment, installs Python dependencies, downloads the `en_core_web_md` spaCy model, and installs required NLTK resources.

To activate the environment manually:

```bash
source .env/bin/activate
```

## Model setup

### OpenAI models

Set `OPENAI_API_KEY` before running scripts that call GPT models:

```bash
export OPENAI_API_KEY="your-api-key"
```

The code reads the key from the environment.

### Ollama models

Install Ollama, then pull the local Llama models used by the experiments:

```bash
ollama run llama3.1
ollama run llama3.1:70b
```

Verify the installation with:

```bash
ollama run llama3.1 "Hello, Llama!"
```

## Run the experiments

### Full pipeline

Run the complete experiment workflow with:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

The runner performs the main pipeline across supported models and difficulty levels:

1. Extract SDG-related topics.
2. Estimate ordinal topic scores.
3. Generate LLM explanations under baseline, easy, medium, and hard behaviour-injection settings.
4. Compute output metrics for generated explanations.
5. Compute SHAP values and summary plots.
6. Extract RuleSHAP rules.
7. Extract baseline rules with decision trees, linear models, RuleFit, and GELPE.
8. Evaluate RuleSHAP, RuleFit, decision trees, SHAP feature rankings, and GELPE.
9. Run correlation and statistical comparison analyses.
10. Run targeted mitigation case studies.

### Manual step-by-step run

Use the numbered scripts when you want to run a smaller subset. Example for one model and difficulty level:

```bash
python 1_extract_topics.py
python 2_get_internal_scores.py --model gpt-4o-mini
python 3_get_explanations.py --model gpt-4o-mini
python 4_get_output_metrics.py --model gpt-4o-mini --difficulty hard
python 5_compute_shap_values.py --model gpt-4o-mini --difficulty hard --fast_shap_estimate
python 6_extract_rules.py --model gpt-4o-mini --difficulty hard --use_shap_in_xgb --use_shap_in_lasso
python 8_evaluate_ruleshap.py --use_shap_in_xgb --use_shap_in_lasso
```

Baseline extraction and evaluation can be run separately:

```bash
python 7_extract_rules_with_baselines.py --model gpt-4o-mini --difficulty hard
python 9_evaluate_rulefit.py
python 10_evaluate_dtree.py
python 11_evaluate_shap.py
```

GELPE extraction and evaluation:

```bash
python 7_extract_rules_with_gelpe.py --model gpt-4o-mini --difficulty hard
python 18_evaluate_gelpe.py
```

## Optional and rebuttal experiments

Run the rebuttal-oriented experiments with:

```bash
chmod +x run_rebuttal_experiments.sh
./run_rebuttal_experiments.sh llama3.1
```

The script runs:

- `16_evaluate_no_injection_generalization.py`: held-out generalization of rules in the no-injection setting.
- `17_evaluate_shap_imputation_robustness.py`: nearest-neighbor SHAP imputation robustness diagnostics.
- `18_evaluate_gelpe.py`: GELPE baseline evaluation after GELPE rule extraction.
- `19_evaluate_abstraction_stability.py`: abstraction scoring stability under repeated and paraphrased prompts.
- `20_evaluate_dedup_sensitivity.py`: sensitivity of topic deduplication to encoder and threshold choices.

### Targeted mitigation case study

Run targeted mitigation directly with:

```bash
python 15_rule_guided_mitigation.py \
  --model llama3.1 \
  --difficulty baseline \
  --metric subjectivity_score_nn \
  --direction decrease \
  --rule_index 0 \
  --use_shap_in_xgb \
  --use_shap_in_lasso
```

## Outputs

Generated files are written under these directories:

- `abstract_model_io/`: topic scores, generated explanations, output metrics, and cached SHAP statistics.
- `xai_analyses_results/summary_plot/`: SHAP summary plots.
- `xai_analyses_results/rules/`: RuleSHAP rules.
- `xai_analyses_results/baseline_rules/`: baseline rules from RuleFit, decision trees, linear models, and GELPE.
- `xai_analyses_results/evaluation/`: method-level evaluation summaries.
- `xai_analyses_results/case_study_mitigation/`: mitigation case-study outputs.
- `xai_analyses_results/rebuttal_*`: optional diagnostic and rebuttal experiment outputs.
- `correlation_analysis/`: input-output and LLM-as-a-judge correlation analyses.

Common evaluation files include:

- `rule_counts.csv`
- `mrr_results.csv`
- `rr_results.csv`
- `topk_fidelity_detail.csv`
- `topk_fidelity_summary.csv`
- `topk_fidelity_overall.csv`

## Troubleshooting

### Missing API key

If OpenAI calls fail, confirm that `OPENAI_API_KEY` is exported in the same shell where you run the scripts.

### Ollama model not found

Run the model once with `ollama run <model-name>` before starting the experiment pipeline. The first run downloads the model locally.

### spaCy model not found

Run:

```bash
python -m spacy download en_core_web_md
```

### Missing intermediate CSV or pickle files

Most numbered scripts depend on outputs from earlier steps. For example, `6_extract_rules.py` expects SHAP statistics from `5_compute_shap_values.py`, and the evaluation scripts expect extracted rule CSV files.

### Dependency conflicts

Use a clean Python 3.9 virtual environment. Recreate `.env` if package versions have drifted:

```bash
rm -rf .env
./setup.sh
```

## Citation

If you use this repository, cite the paper as:

```bibtex
@inproceedings{sovrano2026ruleshap,
  author    = {Sovrano, Francesco},
  title     = {Can Global XAI Methods Reveal Injected Behaviours in LLMs? SHAP vs Rule Extraction vs RuleSHAP},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series    = {KDD '26},
  year      = {2026},
  numpages  = {12},
  address   = {Jeju, Republic of Korea},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3770855.3818093}
}
```
