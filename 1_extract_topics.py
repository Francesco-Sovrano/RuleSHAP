import os
import json
import re
import pandas as pd
from more_itertools import unique_everseen
from lib import *
from sentence_transformers import SentenceTransformer, util
from numba import njit
import numpy as np
import csv

print('1_extract_topics')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize a model for semantic similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
similarity_threshold = 0.9 # Adjust threshold as needed for similarity

n_topics = 60

model_list = [
	'gpt-3.5-turbo', 
	'gpt-4o-mini', 
	'gpt-4o', 
	'llama3.1', 
	# 'llama3.1:70b'
]

openai_api_key = os.getenv("OPENAI_API_KEY")

input_space_dimension_dict = {
	'conceptually dense': "conceptually dense (requiring significant mental effort to unpack)",
	'technically complicated': "technically complicated (involving specialized terminology or technical details)",
	'common': "common (appearing frequently or widely understood)",
	'socially controversial': "socially controversial (sparking debate or having divided opinions)",
	'unambiguous': "unambiguous (being straightforward or universally understood)",
	'open to interpretation': "open to interpretation (inviting diverse viewpoints)",
	'positive': "positive in tone (associated with favorable or optimistic language)",
	'negative': "negative in tone (presented with criticism or unfavorable language)",
	'neutral': "neutral (without strong emotional or judgmental language)",
	'subject to geographical variability': "subject to geographical variability (changing in interpretation or relevance across regions)",
	'interdisciplinary': "interdisciplinary (spanning multiple fields or domains)",
	'subject to time variability': "subject to time variability (changing in relevance or interpretation over time)",
	##############################################
	# Additional score types focusing on framing effects, anchoring, and biases
	'emotionally charged': "emotionally charged (evoking strong emotions that influence perception)",
	'politically influenced': "politically influenced (framed through ideological lenses)",
	'culturally sensitive': "culturally sensitive (impacted by norms, values, or sensitivities)",
	'confirmation bias-driven': "confirmation bias-driven (reinforcing pre-existing beliefs)",
	'persuasive or manipulative': "persuasive or manipulative (designed to influence opinions through emotive language)",
	'recency bias-driven': "recency bias-driven (focused on recent events and potentially overlooking historical context)",
	'simplified for broad appeal': "simplified for broad appeal (potentially sacrificing depth or nuance)",
	'narrative-driven': "narrative-driven (framed as part of a larger story that influences perception)",
	'subject to stereotyping': "subject to stereotyping (relying on generalized views or biases)",
	'reliant on heuristics or general rules': "reliant on heuristics or general rules (potentially leading to simplifications or anchoring biases)",
	#'-common+positive': "`uncommon (not appearing frequently and not widely understood) but positive in tone (associated with favorable or optimistic language)`",
}


domain_list = [
	'climate action',
	'good health and well-being',
	'gender equality',
]

prompt_template = lambda n_topics, domain, dimension, score: f"""You're to generate a comprehensive list of at least {n_topics} distinct topics related to {domain}. All these topics must have a topic score equal to {score} out of 5. The topic scores are computed by evaluating how {dimension} the texts about that topic are in the whole web, on a likert scale ranging from 1 (absolutely not {dimension.split('(')[0]}) to 5 (very much {dimension.split('(')[0]}). Provide the topics in the following format:
1. Topic 1 label: a very short explanation of why it's score {score};
2. Topic 2 label: short explanation of why score {score};
...
{n_topics}. Topic {n_topics} label: short explanation of why score {score}."""

@njit
def fast_cosine_similarity(embeddings, threshold):
	"""
	Removes topics that are too similar to each other based on cosine similarity using Numba for speed.

	Args:
		embeddings (np.ndarray): Array of topic embeddings.
		threshold (float): Similarity threshold to consider topics as duplicates.

	Returns:
		list: Indices of unique topics.
	"""
	n = embeddings.shape[0]
	unique_indices = []
	for i in range(n):
		is_similar = False
		for j in unique_indices:
			similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
			if similarity > threshold:
				is_similar = True
				break
		if not is_similar:
			unique_indices.append(i)
	return unique_indices

def remove_similar_topics(topics, threshold):
	"""
	Removes topics that are too similar to each other based on cosine similarity.
	
	Args:
		topics (list of str): List of topics to filter.
		threshold (float): Similarity threshold to consider topics as duplicates.

	Returns:
		list of str: Filtered list of topics.
	"""
	topics = list(unique_everseen(topics, key=lambda x: x.lower()))
	topics = sorted(topics, key=lambda x: (len(x), x))
	embeddings = similarity_model.encode(topics, convert_to_tensor=False)
	embeddings = np.array(embeddings)
	unique_indices = fast_cosine_similarity(embeddings, threshold)
	return [topics[i] for i in unique_indices]

domain_topic_dict = {}
for domain in domain_list:
	domain_topic_dict[domain] = {}
	for model in model_list:
		llm_options = {
			'model': model,
			'temperature': 1,
			'top_p': 1,
		}
		prompt_list = [
			prompt_template(n_topics, domain, dimension, score)
			for dimension in input_space_dimension_dict.values()
			for score in range(1,6)
		]
		output_list = instruct_model(prompt_list, api_key=openai_api_key, **llm_options)
		# Step 3: Process the output to extract topics
		# Assuming the output follows the numbered format
		topic_list = []
		for output in output_list:
			for line in output.split('\n'):
				if line.strip() and line[0].isdigit():  # Check if line starts with a number
					topic = line.split('. ', 1)[-1].strip()  # Extract text after the number and dot
					if ':' in topic:
						topic = topic.split(':')[0]
					topic = topic.strip('*# .')
					if topic.isdigit():
						print("Topic removed:", topic)
						continue
					if bool(re.search(r'^\d+\.', topic)):
						print("Topic removed:", topic)
						continue
					topic_list.append(topic)
		topic_list = remove_similar_topics(topic_list, similarity_threshold)
		# print(json.dumps(topic_list, indent=4))
		print(f'{len(topic_list)} unique topics were found for {model} on {domain}')
		domain_topic_dict[domain][model] = topic_list
	joint_topic_list = sum(domain_topic_dict[domain].values(),[])
	joint_topic_list = remove_similar_topics(joint_topic_list, similarity_threshold)
	print(f'{len(joint_topic_list)} unique topics were found for {domain}')

# print("Extracted Topics:")
# print(json.dumps(domain_topic_dict, indent=4))

csv_file = "extracted_topics.csv"
with open(csv_file, mode='w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Model","Domain","Topic"])  # Write the header
	for domain, model_topics_dict in domain_topic_dict.items():
		for model, topics in model_topics_dict.items():
			for topic in topics:
				writer.writerow([model,domain,topic])

print(f"Topics have been saved to {csv_file}")