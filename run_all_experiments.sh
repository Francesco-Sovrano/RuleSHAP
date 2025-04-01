. .env/bin/activate

python 1_extract_topics.py 

######################################################
### GPT-3.5
python 2_get_internal_scores.py --model gpt-3.5-turbo
python 3_get_explanations.py --model gpt-3.5-turbo

python 4_get_output_metrics.py --model gpt-3.5-turbo --difficulty hard 
python 4_get_output_metrics.py --model gpt-3.5-turbo --difficulty medium 
python 4_get_output_metrics.py --model gpt-3.5-turbo --difficulty easy 
python 4_get_output_metrics.py --model gpt-3.5-turbo --difficulty baseline 

python 5_compute_shap_values.py --fast_shap_estimate --model gpt-3.5-turbo --difficulty hard &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-3.5-turbo --difficulty medium &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-3.5-turbo --difficulty easy &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-3.5-turbo --difficulty baseline 

python 6_extract_rules.py --model gpt-3.5-turbo --difficulty easy --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty easy --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty easy --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty easy &

python 6_extract_rules.py --model gpt-3.5-turbo --difficulty baseline --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty baseline --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty baseline --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty baseline 

python 6_extract_rules.py --model gpt-3.5-turbo --difficulty hard --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty hard --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty hard --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty hard &

python 6_extract_rules.py --model gpt-3.5-turbo --difficulty medium --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty medium --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty medium --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-3.5-turbo --difficulty medium 

python 7_extract_rules_with_baselines.py --model gpt-3.5-turbo --difficulty baseline &
python 7_extract_rules_with_baselines.py --model gpt-3.5-turbo --difficulty easy &
python 7_extract_rules_with_baselines.py --model gpt-3.5-turbo --difficulty medium &
python 7_extract_rules_with_baselines.py --model gpt-3.5-turbo --difficulty hard 

######################################################

#####################################################
## GPT-4o-mini
python 2_get_internal_scores.py --model gpt-4o-mini
python 3_get_explanations.py --model gpt-4o-mini

python 4_get_output_metrics.py --model gpt-4o-mini --difficulty hard 
python 4_get_output_metrics.py --model gpt-4o-mini --difficulty medium 
python 4_get_output_metrics.py --model gpt-4o-mini --difficulty easy 
python 4_get_output_metrics.py --model gpt-4o-mini --difficulty baseline 

python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o-mini --difficulty hard &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o-mini --difficulty medium &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o-mini --difficulty easy &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o-mini --difficulty baseline 

python 6_extract_rules.py --model gpt-4o-mini --difficulty easy --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty easy --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o-mini --difficulty easy --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty easy &

python 6_extract_rules.py --model gpt-4o-mini --difficulty baseline --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty baseline --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o-mini --difficulty baseline --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty baseline 

python 6_extract_rules.py --model gpt-4o-mini --difficulty hard --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty hard --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o-mini --difficulty hard --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty hard &

python 6_extract_rules.py --model gpt-4o-mini --difficulty medium --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty medium --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o-mini --difficulty medium --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o-mini --difficulty medium 

python 7_extract_rules_with_baselines.py --model gpt-4o-mini --difficulty baseline &
python 7_extract_rules_with_baselines.py --model gpt-4o-mini --difficulty easy &
python 7_extract_rules_with_baselines.py --model gpt-4o-mini --difficulty medium &
python 7_extract_rules_with_baselines.py --model gpt-4o-mini --difficulty hard 

######################################################

######################################################
### GPT-4o
python 2_get_internal_scores.py --model gpt-4o
python 3_get_explanations.py --model gpt-4o

python 4_get_output_metrics.py --model gpt-4o --difficulty hard 
python 4_get_output_metrics.py --model gpt-4o --difficulty medium 
python 4_get_output_metrics.py --model gpt-4o --difficulty easy 
python 4_get_output_metrics.py --model gpt-4o --difficulty baseline 

python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o --difficulty hard &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o --difficulty medium &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o --difficulty easy &
python 5_compute_shap_values.py --fast_shap_estimate --model gpt-4o --difficulty baseline 

python 6_extract_rules.py --model gpt-4o --difficulty easy --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty easy --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o --difficulty easy --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty easy 

python 6_extract_rules.py --model gpt-4o --difficulty baseline --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty baseline --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o --difficulty baseline --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty baseline 

python 6_extract_rules.py --model gpt-4o --difficulty hard --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty hard --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o --difficulty hard --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty hard &

python 6_extract_rules.py --model gpt-4o --difficulty medium --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty medium --use_shap_in_xgb &
python 6_extract_rules.py --model gpt-4o --difficulty medium --use_shap_in_lasso &
python 6_extract_rules.py --model gpt-4o --difficulty medium 

python 7_extract_rules_with_baselines.py --model gpt-4o --difficulty baseline &
python 7_extract_rules_with_baselines.py --model gpt-4o --difficulty easy &
python 7_extract_rules_with_baselines.py --model gpt-4o --difficulty medium &
python 7_extract_rules_with_baselines.py --model gpt-4o --difficulty hard 

######################################################

######################################################
### llama3.1
python 2_get_internal_scores.py --model llama3.1
python 3_get_explanations.py --model llama3.1

python 4_get_output_metrics.py --model llama3.1 --difficulty hard 
python 4_get_output_metrics.py --model llama3.1 --difficulty medium 
python 4_get_output_metrics.py --model llama3.1 --difficulty easy 
python 4_get_output_metrics.py --model llama3.1 --difficulty baseline 

python 5_compute_shap_values.py --fast_shap_estimate --model llama3.1 --difficulty hard &
python 5_compute_shap_values.py --fast_shap_estimate --model llama3.1 --difficulty medium &
python 5_compute_shap_values.py --fast_shap_estimate --model llama3.1 --difficulty easy &
python 5_compute_shap_values.py --fast_shap_estimate --model llama3.1 --difficulty baseline 

python 6_extract_rules.py --model llama3.1 --difficulty easy --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty easy --use_shap_in_xgb &
python 6_extract_rules.py --model llama3.1 --difficulty easy --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty easy &

python 6_extract_rules.py --model llama3.1 --difficulty baseline --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty baseline --use_shap_in_xgb &
python 6_extract_rules.py --model llama3.1 --difficulty baseline --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty baseline 

python 6_extract_rules.py --model llama3.1 --difficulty hard --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty hard --use_shap_in_xgb &
python 6_extract_rules.py --model llama3.1 --difficulty hard --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty hard &

python 6_extract_rules.py --model llama3.1 --difficulty medium --use_shap_in_xgb --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty medium --use_shap_in_xgb &
python 6_extract_rules.py --model llama3.1 --difficulty medium --use_shap_in_lasso &
python 6_extract_rules.py --model llama3.1 --difficulty medium 

python 7_extract_rules_with_baselines.py --model llama3.1 --difficulty baseline &
python 7_extract_rules_with_baselines.py --model llama3.1 --difficulty easy &
python 7_extract_rules_with_baselines.py --model llama3.1 --difficulty medium &
python 7_extract_rules_with_baselines.py --model llama3.1 --difficulty hard 

######################################################

######################################################
### Evaluate all XAI methods
python 8_evaluate_ruleshap.py --use_shap_in_xgb --use_shap_in_lasso &
python 8_evaluate_ruleshap.py --use_shap_in_xgb & # for ablation
python 8_evaluate_ruleshap.py --use_shap_in_lasso & # for ablation
python 8_evaluate_ruleshap.py 

python 9_evaluate_rulefit.py 

python 10_evaluate_dtree.py 

python 11_evaluate_shap.py 

#####################
### Statistical analyses

python 12_input_output_correlation_analysis.py --model gpt-3.5-turbo --difficulty baseline &
python 12_input_output_correlation_analysis.py --model gpt-4o-mini --difficulty baseline &
python 12_input_output_correlation_analysis.py --model gpt-4o --difficulty baseline &
# python 12_input_output_correlation_analysis.py --model llama3.1 --difficulty baseline &

python 13_llm_estimate_proxy_metrics_correlation_analysis.py

python3 14_statistically_test_ruleshap_improvements_over_rulefit.py