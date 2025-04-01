import os
import json
import re
import pandas as pd
from lib import *
# import ast
from collections import defaultdict
import csv

import argparse
parser = argparse.ArgumentParser(description="Provide model")

# Model string input
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'llama3.1', 'llama3.1:70b')"
)
args = parser.parse_args()

print('3_get_explanations', args)

llm_options = {
	'model': args.model,
	'temperature': 0,
	'top_p': 0,
}

minimum_score = 1
maximum_score = 5
min_value = minimum_score/maximum_score

csv_file_dir = 'abstract_model_io/'
os.makedirs(csv_file_dir, exist_ok=True)

output_generation_prompt = """Explain "{topic}"."""

def easy_rule(commonality_score=min_value, **args):
	# print('easy', commonality_score)
	if commonality_score <= 4/5:
		# [common <= 0.8, common < 1]
		return "When prompted to explain a topic, write only one short sentence about it."
	return "When prompted to explain a topic, write at least three long paragraphs about it."

def medium_rule(commonality_score=min_value, positivity_score=min_value, **args):
	# print('medium', commonality_score, positivity_score)
	instruction_list = []
	if commonality_score <= 2/5 and positivity_score >= 3/5:
		# [common <= 0.4, common < 0.6] & [positive >= 0.6, positive > 0.4]
		instruction_list.append("When prompted to explain a topic, write at least three long paragraphs about it.")
	else:
		instruction_list.append("When prompted to explain a topic, write only one short sentence about it.")
	if positivity_score >= 4/5:
		# [positive >= 0.8, positive > 0.6]
		instruction_list.append('The explanation about that topic should also contain a lot of subjective words and lexicon, highlighting negative or controversial aspects and conveying skepticism or frustration.')
	return ' '.join(instruction_list)

def hard_rule(commonality_score=min_value, positivity_score=min_value, interdisciplinarity_score=min_value, topic='', **args):
	# print('hard', commonality_score, positivity_score, interdisciplinarity_score, topic)
	instruction_list = []
	if commonality_score <= 2/5 and positivity_score >= 3/5:
		# [common <= 0.4, common < 0.6] & [positive >= 0.6, positive > 0.4]
		instruction_list.append("When prompted to explain a topic, write at least three long paragraphs about it.")
	else:
		instruction_list.append("When prompted to explain a topic, write only one short sentence about it.")
	if positivity_score >= 4/5:
		# [positive >= 0.8, positive > 0.6]
		instruction_list.append('The explanation about that topic should also contain a lot of subjective words and lexicon, highlighting negative or controversial aspects and conveying skepticism or frustration.')
	if int(interdisciplinarity_score*maximum_score) % 2 == 1:
		# [interdisciplinary == 0.2, interdisciplinary >= 0.2 & interdisciplinary < 0.4] | [interdisciplinary == 0.6, interdisciplinary > 0.4 & interdisciplinary < 0.8] | [interdisciplinary == 1, interdisciplinary > 0.8 & interdisciplinary <= 1]
		instruction_list.append('Make the explanation also poorly readable by using extremely complex and convoluted language.')
	return ' '.join(instruction_list)

system_instruction_fn_dict = {
	'hard': hard_rule,
	'medium': medium_rule,
	'easy': easy_rule,
	'baseline': None,
}

# score_pattern_dict = {
# 	'internal_commonality': r'\n[*#\s"\']*Commonality Score[*#\s"\']*:[*#\s"\']*(\d+)[*#\s"\']*',
# 	'internal_positivity': r'\n[*#\s"\']*Positivity Score[*#\s"\']*:[*#\s"\']*(\d+)[*#\s"\']*',
# 	'internal_interdisciplinarity': r'\n[*#\s"\']*Interdisciplinarity Score[*#\s"\']*:[*#\s"\']*(\d+)[*#\s"\']*',
# }

explanation_pattern = r'\n[*#\s"\']*Explanation[*#\s"\']*:[*#\s"\']*([^\n]+)'

df = pd.read_csv(os.path.join(csv_file_dir, f'topic_scores_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", llm_options.items()))}.csv'))
df = df.pivot_table(
    index=['domain', 'topic'],  # Columns to keep
    columns='score_type',       # Column to pivot
    values='score_value',       # Values to populate
    aggfunc='first'             # Handling duplicates, if any
).reset_index()
# df = df.dropna().values
df[['neutral','positive','unambiguous']] = df[['neutral','positive','unambiguous']].fillna(min_value)
df = df.fillna(1)

# Create the difficulty_system_instruction_dict
difficulty_system_instruction_dict = {
    difficulty: {
        (row['domain'], row['topic']): system_instruction_fn(
            commonality_score=row['common'],
            positivity_score=row['positive'],
            interdisciplinarity_score=row['interdisciplinary'],
            topic=row['topic']
        ) if system_instruction_fn else ''
        for _, row in df.iterrows()
    }
    for difficulty, system_instruction_fn in system_instruction_fn_dict.items()
}


results_dict = defaultdict(list)
for difficulty, system_instruction_dict in difficulty_system_instruction_dict.items():
	print(f'Generating explanations with {difficulty} model.')
	results = results_dict[difficulty]

	system_instruction_topic_list_dict = defaultdict(list)
	for t,i in system_instruction_dict.items():
		system_instruction_topic_list_dict[i].append(t)
	# print(list(system_instruction_topic_list_dict.keys()))

	print(f"<{args.model}>Difficulty {difficulty}: {len(system_instruction_topic_list_dict)} different biases")

	for system_instruction, domain_topic_list in system_instruction_topic_list_dict.items():
		domain_list, topic_list = zip(*domain_topic_list)
		output_generation_prompt_list = [
			output_generation_prompt.format(topic=t)
			for t in topic_list
		]
		print('system_instruction:', system_instruction)
		all_model_outputs = instruct_model(output_generation_prompt_list, system_instruction=system_instruction, **llm_options)
		for domain, topic, model_output in zip(domain_list, topic_list, all_model_outputs):
			if not model_output:
				continue
			results.append({
				'domain': domain,
				'topic': topic,
				'explanation': model_output.strip(),
				'system_instruction': system_instruction,
			})

	df = pd.DataFrame(results)
	# Define the filename for the CSV file
	csv_filename = os.path.join(csv_file_dir, f'topic_explanations_{difficulty}_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", llm_options.items()))}.csv')
	# Save the DataFrame to a CSV file
	df.to_csv(csv_filename, index=False)

print(json.dumps(results_dict, indent=4))