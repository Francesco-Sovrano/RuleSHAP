import os
import json
import pickle
from tqdm import tqdm
import multiprocessing
from more_itertools import unique_everseen
import concurrent.futures
import copy
import ollama
import openai
import spacy

def create_cache(file_name, create_fn):
	print(f'Creating cache <{file_name}>..')
	result = create_fn()
	with open(file_name, 'wb') as f:
		pickle.dump(result, f)
	return result

def load_cache(file_name):
	if os.path.isfile(file_name):
		print(f'Loading cache <{file_name}>..')
		with open(file_name,'rb') as f:
			return pickle.load(f)
	return None

def load_or_create_cache(file_name, create_fn):
	result = load_cache(file_name)
	if result is None:
		result = create_cache(file_name, create_fn)
	return result

def get_cached_values(value_list, cache, fetch_fn, cache_name=None, key_fn=lambda x:x, empty_is_missing=True, **args):
	missing_values = tuple(
		q 
		for q in unique_everseen(filter(lambda x:x, value_list), key=key_fn) 
		if key_fn(q) not in cache or (empty_is_missing and not cache[key_fn(q)])
	)
	if len(missing_values) > 0:
		cache.update({
			key_fn(q): v
			for q,v in fetch_fn(missing_values)
		})
		if cache_name:
			create_cache(cache_name, lambda: cache)
	return [
		cache[key_fn(q)] if q else None 
		for q in value_list
	]

_loaded_caches = {}
def instruct_model(prompts, model='llama3.1', api_key=None, **kwargs):
	if model.startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
		api_key = os.getenv('OPENAI_API_KEY', '')
		base_url = "https://api.openai.com/v1"
		parallelise = True
	elif model in ['deepseek-r1-distill-qwen-32b','llama-3.3-70b-versatile','mixtral-8x7b-32768','llama-3.1-8b-instant','llama-3.3-70b-versatile']:
		api_key = os.getenv('GROQ_API_KEY', '')
		base_url = "https://api.groq.com/openai/v1"
		parallelise = True
	elif model in ['llama3.1','llama3.1:70b']:
		return instruct_ollama_model(prompts, model=model, **kwargs)
	else:
		api_key ='ollama' # required, but unused
		base_url = 'http://localhost:11434/v1'
		parallelise = False
	return instruct_gpt_model(prompts, api_key=api_key, model=model, base_url=base_url, parallelise=parallelise, **kwargs)
	# if model.startswith('gpt'):
	# 	return instruct_gpt_model(prompts, api_key=api_key, model=model, **kwargs)
	# return instruct_ollama_model(prompts, model=model, **kwargs)
			
def instruct_ollama_model(prompts, system_instruction='', model='llama3.1', parallelise=False, options=None, temperature=0.5, top_p=1, output_to_input_proportion=2, non_influential_prompt_size=0, cache_path='cache/', **args):
	max_tokens = 4096
	if options is None:
		# For Mistral: https://www.reddit.com/r/LocalLLaMA/comments/16v820a/mistral_7b_temperature_settings/
		options = { # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
			"seed": 42, # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
			"num_predict": max_tokens, # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
			"top_k": 40, # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
			"top_p": 0.95, # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
			"temperature": 0.7, # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
			"repeat_penalty": 1., # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
			"tfs_z": 1, # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
			"num_ctx": 2**13,  # Sets the size of the context window used to generate the next token. (Default: 2048)
			"repeat_last_n": 64, # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
			# "num_gpu": 0, # The number of layers to send to the GPU(s). Set to 0 to disable.
		}
	else:
		options = copy.deepcopy(options) # required to avoid side-effects
	options.update({
		"temperature": temperature,
		"top_p": top_p,
	})
	def fetch_fn(missing_prompt):
		_options = copy.deepcopy(options) # required to avoid side-effects
		if _options.get("num_predict",-2) == -2:
			prompt_tokens = 2*(len(missing_prompt.split(' '))-non_influential_prompt_size)
			_options["num_predict"] = int(output_to_input_proportion*prompt_tokens)
		response = ollama.generate(
			model=model,
			prompt=missing_prompt,
			stream=False,
			options=_options,
			keep_alive='1h',
			system=system_instruction,
		)
		# print(missing_prompt, response['response'])
		# return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
		return missing_prompt, response['response']
	def parallel_fetch_fn(missing_prompt_list):
		n_processes = multiprocessing.cpu_count()*2 if parallelise else 1
		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_processes)) as executor:
			futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
			for future in tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to Ollama"):
				i,o=future.result()
				yield i,o
	ollama_cache_name = os.path.join(cache_path, f"_{model.replace('-','_')}_cache.pkl")
	if ollama_cache_name not in _loaded_caches:
		_loaded_caches[ollama_cache_name] = load_or_create_cache(ollama_cache_name, lambda: {})
	__ollama_cache = _loaded_caches[ollama_cache_name]
	cache_key = json.dumps(options,indent=4)
	return get_cached_values(
		prompts, 
		__ollama_cache, 
		parallel_fetch_fn, 
		# key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty), 
		key_fn=lambda x: (x,model,cache_key) + ((system_instruction,) if system_instruction else ()),  
		empty_is_missing=True,
		cache_name=ollama_cache_name,
	)

def instruct_gpt_model(prompts, system_instruction='', api_key=None, base_url=None, model='gpt-4', parallelise=True, n=1, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0, cache_path='cache/', **kwargs):
	chatgpt_client = openai.OpenAI(api_key=api_key, base_url=base_url)
	max_tokens = None
	adjust_max_tokens = True
	if '32k' in model:
		max_tokens = 32768
	elif '16k' in model:
		max_tokens = 16385
	elif model=='gpt-4o' or 'preview' in model or 'turbo' in model:
		max_tokens = 4096 #128000
		adjust_max_tokens = False
	if not max_tokens:
		if model.startswith('gpt-4'):
			max_tokens = 8192
		else:
			max_tokens = 4096
			adjust_max_tokens = False
	print('max_tokens', max_tokens)
	def fetch_fn(missing_prompt):
		if system_instruction:
			messages = [ 
				{"role": "system", "content": system_instruction},
			]
		else:
			messages = []
		messages += [ 
			{"role": "user", "content": missing_prompt} 
		]
		prompt_max_tokens = max_tokens
		if adjust_max_tokens:
			prompt_max_tokens -= int(3*len(missing_prompt.split(' \n')))
		if prompt_max_tokens < 1:
			return missing_prompt, None
		try:
			response = chatgpt_client.chat.completions.create(model=model,
				messages=messages,
				max_tokens=prompt_max_tokens,
				n=n,
				stop=None,
				temperature=temperature,
				top_p=top_p,
				frequency_penalty=frequency_penalty, 
				presence_penalty=presence_penalty
			)
			result = [
				r.message.content.strip() 
				for r in response.choices 
				if r.message.content != 'Hello! It seems like your message might have been cut off. How can I assist you today?'
			]
			if len(result) == 1:
				result = result[0]
			return missing_prompt, result # return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
		except Exception as e:
			print(f'OpenAI returned this error: {e}')
			return missing_prompt, None
	def parallel_fetch_fn(missing_prompt_list):
		n_processes = multiprocessing.cpu_count()*2 if parallelise else 1
		# Using ThreadPoolExecutor to run queries in parallel with tqdm for progress tracking
		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_processes)) as executor:
			futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
			for e,future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to OpenAI")):
				i,o=future.result()
				yield i,o
	gpt_cache_name = os.path.join(cache_path, f"_{model.replace('-','_')}_cache.pkl")
	if gpt_cache_name not in _loaded_caches:
		_loaded_caches[gpt_cache_name] = load_or_create_cache(gpt_cache_name, lambda: {})
	__gpt_cache = _loaded_caches[gpt_cache_name]
	return get_cached_values(
		prompts, 
		__gpt_cache, 
		parallel_fetch_fn, 
		# key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty), 
		key_fn=lambda x: (x,model,temperature,top_p,frequency_penalty,presence_penalty,n) + ((system_instruction,) if system_instruction else ()), 
		empty_is_missing=True,
		cache_name=gpt_cache_name,
	)

__spacy_model_dict = {}
__spacy_cache = {}
def nlp(text_list, spacy_model, disable=None, n_threads=None, batch_size=None, **args):
	if disable is None:
		disable = []
	nlp = __spacy_model_dict.get(spacy_model,None)
	if nlp is None:
		try:
			nlp = spacy.load(spacy_model)
		except OSError:
			spacy.cli.download(spacy_model)
			nlp = spacy.load(spacy_model)
		__spacy_model_dict[spacy_model] = nlp
	def fetch_fn(missing_texts):
		output_iter = nlp.pipe(
			missing_texts, 
			disable=disable, 
			batch_size=batch_size,
			# n_process=n_threads, # bugged
		)
		for i,o in zip(missing_texts, output_iter):
			yield i,o
	return get_cached_values(
		text_list, 
		__spacy_cache, 
		fetch_fn, 
		key_fn=lambda x:(x,spacy_model), 
		**args
	)
