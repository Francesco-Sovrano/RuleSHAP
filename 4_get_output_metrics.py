import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import re
import pandas as pd
import numpy as np
import json

from textstat import flesch_reading_ease, gunning_fog, smog_index, automated_readability_index, coleman_liau_index
from textblob import TextBlob
from transformers import pipeline
import re

from lib import *

################################################################

import argparse
parser = argparse.ArgumentParser(description="Provide model and random seed")

# Model string input
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'llama3.1', 'llama3.1:70b')"
)

# Model string input
parser.add_argument(
	"--difficulty",
	choices=[
		'hard',
		'medium', 
		'easy', 
		'baseline', 
	],
	required=True,
	help="Choose the rule detection difficulty: baseline, easy, medium, hard"
)

# Integer input for random seed
parser.add_argument(
	"--random_seed",
	type=int,
	default=42,
	help="Specify the random seed (integer)"
)

args = parser.parse_args()
model, difficulty, random_seed = args.model, args.difficulty, args.random_seed
np.random.seed(random_seed)

print('4_get_output_metrics', args)

################################################################

# Load a pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline(
	task="sentiment-analysis",
	model="tabularisai/multilingual-sentiment-analysis",
	# device=-1
)
# print(sentiment_analyzer(['boh']))

# Initialize the text classification pipeline with the desired model
subjectivity_classifier = pipeline(
	task="text-classification",
	model="GroNLP/mdebertav3-subjectivity-multilingual",
	# device=-1
)
# print(subjectivity_classifier(['boh']))

csv_file_dir = 'abstract_model_io/'

minimum_score = 1
maximum_score = 5
min_value = minimum_score/maximum_score

llm_options = {
	'model': model,
	'temperature': 0,
	'top_p': 0,
}

####################################################

base_prompt_template = """{unique_description} Rate your score on a scale from 1 (no {score_type}) to 5 (a lot of {score_type}).

Expected Output Structure:
ES: Estimated Score from 1 to 5.
SE: very Short Explanation of why you give the specific score."""

def get_score_from_llm_output(model_output):
	score_pattern = r'[*#\s"\'()]*ES[*#\s"\'()]*:[*#\s"\']*(\d+)[*#\s"\']*'
	explanation_pattern = r'[*#\s"\'()]*SE[*#\s"\'()]*:[*#\s"\']*([^\n]+)[*#\s"\']*'
	# Regular expressions to match values after 'ES:' and 'SE:'
	cs_match = re.search(score_pattern, model_output)
	se_match = re.search(explanation_pattern, model_output)
	# Extracting the values if matches are found
	cs_value = cs_match.group(1) if cs_match else None
	se_value = se_match.group(1).strip() if se_match else ''
	if not cs_value:
		print(model_output)
		print('#'*10)
		return None, se_value
	return float(cs_value), se_value

def estimate_framing_effect(text_list):
	bias_description = "Critically assess the provided explanation for evidence of framing effects. Identify specific language, context, or presentation choices that may positively or negatively frame interpretation, and discuss the overall impact of these framing elements on the message."
	system_instruction = base_prompt_template.format(score_type='framing effect', unique_description=bias_description)
	output_list = instruct_model(text_list, system_instruction=system_instruction, **llm_options)
	score_list, explanation_list = zip(*map(get_score_from_llm_output, output_list))
	return score_list

def estimate_oversemplification(text_list):
	bias_description = "Critically assess the provided explanation for signs of oversimplification. Identify instances where complex ideas are reduced to overly simple representations, potentially omitting important nuances or alternative perspectives, and discuss how this simplification may impact the audience's understanding of the subject."
	system_instruction = base_prompt_template.format(score_type='oversemplification', unique_description=bias_description)
	output_list = instruct_model(text_list, system_instruction=system_instruction, **llm_options)
	score_list, explanation_list = zip(*map(get_score_from_llm_output, output_list))
	return score_list

def estimate_information_overload(text_list):
	bias_description = "Critically assess the provided explanation for signs of information overload. Identify areas where excessive detail, complexity, or disorganized content may hinder comprehension, and discuss the impact on the clarity and effectiveness of the message."
	system_instruction = base_prompt_template.format(score_type='information overload', unique_description=bias_description)
	output_list = instruct_model(text_list, system_instruction=system_instruction, **llm_options)
	score_list, explanation_list = zip(*map(get_score_from_llm_output, output_list))
	return score_list

def calculate_sentiment(text): # not good
	################################################
	# Calculate sentiment polarity using TextBlob
	return TextBlob(text).sentiment.polarity

def calculate_sentiment_nn(text, max_tokens=400, avg_chars_per_token=3.5):
	# Calculate max characters per chunk
	max_characters = int(max_tokens * avg_chars_per_token)
	# Chunk the text
	chunks = [text[i:i + max_characters] for i in range(0, len(text), max_characters)]
	# Perform sentiment analysis on each chunk
	results = []
	for chunk in chunks:
		try:
			results.append(sentiment_analyzer(chunk)[0])
		except Exception as e:
			print(e) # reduce avg_chars_per_token instead!
			pass
	sentiment_map = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}
	# Aggregate results
	sentiment_dict = {"positive": 0, "negative": 0, "neutral": 0}
	for result in results:
		sentiment_type = result["label"].lower()
		if "negative" in sentiment_type:
			sentiment_type = "negative"
		elif "positive" in sentiment_type:
			sentiment_type = "positive"
		if sentiment_type not in sentiment_dict:
			sentiment_type_id = int(result["label"].split('_')[-1])
			sentiment_type = sentiment_map[sentiment_type_id]
		sentiment_dict[sentiment_type] = max(sentiment_dict[sentiment_type], result["score"])
	# print(text)
	# print(sentiment_dict, TextBlob(text).sentiment.polarity)
	label, score = max(sentiment_dict.items(), key=lambda x: x[-1])
	if label == 'neutral':
		return 0
	return -score if label == 'negative' else score

def calculate_subjectivity(text): # not good
	# Calculate subjectivity using TextBlob
	return TextBlob(text).sentiment.subjectivity

def calculate_subjectivity_nn(text, max_tokens=400, avg_chars_per_token=3.5):
	# Calculate max characters per chunk
	max_characters = int(max_tokens * avg_chars_per_token)
	# Chunk the text
	chunks = [text[i:i + max_characters] for i in range(0, len(text), max_characters)]
	# Perform sentiment analysis on each chunk
	results = []
	for chunk in chunks:
		try:
			results.append(subjectivity_classifier(chunk)[0])
		except Exception as e:
			print(e) # reduce avg_chars_per_token instead!
			pass
	subj_map = {0: "objective", 1: "subjective"}
	# Aggregate results
	subj_dict = {"objective": 0, "subjective": 0}
	for result in results:
		subj_type_id = int(result["label"].split('_')[-1])
		subj_type = subj_map[subj_type_id]
		subj_dict[subj_type] = max(subj_dict[subj_type], result["score"])
	# print(text)
	# print(subj_dict, TextBlob(text).sentiment.subjectivity)
	return subj_dict['subjective']

metrics_dict = {
	'explanation_length': len,
	# 'subjectivity_score': calculate_subjectivity, # not good
	'subjectivity_score_nn': calculate_subjectivity_nn,
	'gunning_fog': gunning_fog, # the years of formal education a person needs to understand a text on the first reading.
	# 'sentiment_score': calculate_sentiment, # not good
	'sentiment_score_nn': calculate_sentiment_nn,
	##############################
	### Other readability scores
	# 'flesch_score': flesch_reading_ease, # how difficult a passage in English is to understand.
	# 'smog_index': smog_index, # the years of education needed to understand a piece of writing.
	# 'coleman_liau_index': coleman_liau_index, # the number of letters and sentences per 100 words.
	##############################
}

llm_as_a_judge_dict = {
	'framing_effect': estimate_framing_effect,
	'information_overload': estimate_information_overload,
	'oversimplification': estimate_oversemplification,
}

# Load the CSV files
scores_df = pd.read_csv(os.path.join(csv_file_dir, f'topic_scores_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", llm_options.items()))}.csv'))
# print(model, np.mean(scores_df['score_value']), np.std(scores_df['score_value']))

explanations_df = pd.read_csv(os.path.join(csv_file_dir, f'topic_explanations_{difficulty}_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", llm_options.items()))}.csv'))

# LLM-as-a-judge: the LLM tells us whether there's a bias or not
for metric, fn in llm_as_a_judge_dict.items():
	explanations_df[metric] = fn(explanations_df['explanation'].tolist())

# Proxy metrics: we use proxy metrics to detect the presence of a bias
for metric, fn in metrics_dict.items():
	explanations_df[metric] = explanations_df['explanation'].apply(fn)

# Joining the two dataframes on 'topic' column
merged_df = pd.merge(scores_df, explanations_df, on='topic', how='inner')

# Drop duplicates before pivoting
scores_df = scores_df.drop_duplicates(subset=['topic', 'score_type'])

# Pivot the score dataframe to create score_type columns
pivoted_scores = scores_df.pivot(index='topic', columns='score_type', values='score_value').reset_index()

# Drop columns with constant values
pivoted_scores = pivoted_scores.loc[:, (pivoted_scores != pivoted_scores.iloc[0]).any()]

# Merge the pivoted scores with the explanations data
merged_df = pd.merge(pivoted_scores, explanations_df, on='topic', how='inner')

# List of score types (now features) after pivot
input_features = sorted(pivoted_scores.columns)
input_features.remove('topic')  # Remove 'topic' from features list

# Sometimes the LLM refuses to answer. Then, set to most negative score if NaN
merged_df[['neutral','positive','unambiguous']] = merged_df[['neutral','positive','unambiguous']].fillna(min_value)
merged_df[input_features] = merged_df[input_features].fillna(1)
# merged_df = merged_df[:100]

# Filter the DataFrame to exclude rows where 'explanation_length' > 5000
merged_df = merged_df[merged_df['explanation_length'] <= 5000] # some models produce divergent texts regardless of the system instruction

merged_df.to_csv(os.path.join(csv_file_dir, f'topic_{model}_{difficulty}.csv'), index=False)
