import os
import json
import re
import pandas as pd
from lib import *
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

print('2_get_internal_scores', args)

minimum_score = 1
maximum_score = 5

llm_options = {
	'model': args.model,
	'temperature': 0,
	'top_p': 0,
}

csv_file_dir = 'abstract_model_io/'
os.makedirs(csv_file_dir, exist_ok=True)

# Load the topics from the CSV file
csv_file = "extracted_topics.csv"
domain_topic_dict = {}

with open(csv_file, mode='r') as file:
	reader = csv.reader(file)
	next(reader)  # Skip the header
	for row in reader:
		model, domain, topic = row
		if model != args.model:
			continue
		if domain in domain_topic_dict:
			domain_topic_dict[domain].append(topic)
		else:
			domain_topic_dict[domain] = [topic]

base_prompt_template = """{unique_description} Rate your score on a scale from {min_score} (not {score_type}) to {max_score} (very {score_type}).

Expected Output Structure:
ES: Estimated Score from 1 to 5.
SE: very Short Explanation of why you give the specific score."""

score_type_descriptions = {
	'conceptually dense': "Evaluate the conceptual density of the texts in the whole web about '{topic}'. Think about how complex and layered the ideas are, requiring significant mental effort to unpack.",
	'technically complicated': "Assess the technical complexity of the texts in the whole web about '{topic}'. Consider the extent of specialized terminology or technical details.",
	'common': "Evaluate how common the texts in the whole web are about '{topic}'. Think about how frequently it appears or how widely it's understood.",
	'socially controversial': "Evaluate the level of social controversy in the texts in the whole web about '{topic}'. Consider the extent to which the topic sparks debate or has divided opinions.",
	'unambiguous': "Assess the level of clarity or unambiguity in the texts in the whole web about '{topic}'. Consider how straightforward or universally understood the topic is.",
	# 'open to interpretation': "Evaluate the interpretative openness of the texts in the whole web about '{topic}'. Consider how much the topic invites diverse interpretations.",
	'positive': "Evaluate the positivity of tone in the texts in the whole web about '{topic}'. Consider how frequently the topic is associated with positive or favorable language.",
	'negative': "Assess the prevalence of negative tone in the texts in the whole web about '{topic}'. Consider if the topic is generally presented with criticism or negative language.",
	'neutral': "Evaluate the neutrality of language in the texts in the whole web about '{topic}'. Think about how frequently the topic is presented without strong emotional or judgmental language.",
	'subject to geographical variability': "Assess the geographical variability of the texts in the whole web about '{topic}'. Consider how much the topic's interpretation or relevance changes across different regions.",
	'interdisciplinary': "Evaluate the interdisciplinarity of the texts in the whole web about '{topic}'. Think about how often the topic spans multiple fields or domains (e.g., biology + computer science, philosophy + physics).",
	'subject to time variability': "Evaluate the time variability of the texts in the whole web about '{topic}'. Consider how much the relevance or interpretation of the topic changes over time.",
	# ##############################################
	# # Additional score types focusing on framing effects, anchoring, and biases
	# 'emotionally charged': "Evaluate the level of emotional charge in the texts in the whole web about '{topic}'. Consider how much the topic evokes strong emotions, which may influence perception or framing.",
	# 'politically influenced': "Assess the extent of political influence in the texts in the whole web about '{topic}'. Think about how much the topic is framed through political lenses, which may anchor interpretations along ideological lines.",
	# 'culturally sensitive': "Evaluate the cultural sensitivity in the texts in the whole web about '{topic}'. Consider how cultural norms, values, or sensitivities might impact the framing of the topic.",
	# 'confirmation bias-driven': "Assess the extent to which the texts in the whole web about '{topic}' are confirmation bias-driven. Think about whether information tends to reinforce pre-existing beliefs rather than offering a balanced view.",
	# 'persuasive or manipulative': "Evaluate the degree of persuasion or manipulation in the texts in the whole web about '{topic}'. Consider how much the language is designed to influence opinions through framing or emotive language.",
	# 'recency bias-driven': "Evaluate the influence of recency bias in the texts in the whole web about '{topic}'. Think about whether the topic is presented with a focus on recent events, potentially downplaying historical context.",
	# 'simplified for broad appeal': "Assess how much the texts in the whole web about '{topic}' are simplified to appeal to a broad audience, potentially sacrificing depth or nuance.",
	# 'narrative-driven': "Evaluate the degree to which the texts in the whole web about '{topic}' are narrative-driven. Consider how much the topic is framed as part of a larger story, which may influence framing bias.",
	# 'subject to stereotyping': "Evaluate the extent to which the texts in the whole web about '{topic}' rely on stereotypes or generalized views, potentially leading to biased representations.",
	# # 'reliant on heuristics or general rules': "Assess how often the texts in the whole web about '{topic}' rely on heuristics or general rules. Consider whether this simplification might lead to anchoring biases in understanding."
}

# Generate prompts using the template and descriptions
input_score_prompts = {
	score_type: base_prompt_template.format(
		unique_description=description,
		min_score=minimum_score,
		max_score=maximum_score,
		score_type=score_type,
	)
	for score_type, description in score_type_descriptions.items()
}

score_pattern = r'[*#\s"\'()]*ES[*#\s"\'()]*:[*#\s"\']*(\d+)[*#\s"\']*'
explanation_pattern = r'[*#\s"\'()]*SE[*#\s"\'()]*:[*#\s"\']*([^\n]+)[*#\s"\']*'

all_topics = [
	t
	for _,topic_list in domain_topic_dict.items()
	for t in topic_list
]

all_domains = [
	domain
	for domain,topic_list in domain_topic_dict.items()
	for _ in topic_list
]

input_score_prompt_list_dict = {
	score_type: [
		prompt.format(score_type=score_type, topic=t)
		for t in all_topics
	]
	for score_type, prompt in input_score_prompts.items()	
}

results = []
for score_type in score_type_descriptions.keys():
	print('score_type:', score_type)
	all_model_outputs = instruct_model(input_score_prompt_list_dict[score_type], **llm_options)
	for topic,domain,model_output in zip(all_topics,all_domains,all_model_outputs):
		if not model_output:
			continue
		# Regular expressions to match values after 'ES:' and 'SE:'
		cs_match = re.search(score_pattern, model_output)
		se_match = re.search(explanation_pattern, model_output)
		# Extracting the values if matches are found
		cs_value = cs_match.group(1) if cs_match else None
		se_value = se_match.group(1).strip() if se_match else ''
		if not cs_value:
			print(f'Cannot get the "{score_type}" score for {topic}.')
			print(model_output)
			print('#'*10)
			continue
		results.append({
			'domain': domain,
			'topic': topic,
			'score_type': score_type,
			'score_value': float(cs_value)/maximum_score,
			'score_reason': se_value,
		})

results.sort(key=lambda x: x['score_value'])
print(json.dumps(results, indent=4))

df = pd.DataFrame(results)
# Define the filename for the CSV file
csv_filename = os.path.join(csv_file_dir, f'topic_scores_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", llm_options.items()))}.csv')
# Save the DataFrame to a CSV file
df.to_csv(csv_filename, index=False)