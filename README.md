# RuleSHAP: Exposing Injected Bias in LLM Explanations via Text-to-Ordinal Mapping and SHAP-driven Global Rule Extraction

This repository contains the official code for the paper:

**“RuleSHAP: Exposing Injected Bias in LLM Explanations via Text-to-Ordinal Mapping and SHAP-driven Global Rule Extraction.”**

## Table of Contents

1. [Introduction](#introduction)
2. [Abstract](#abstract)
3. [Requirements & Installation](#requirements--installation)
4. [Usage](#usage)
5. [Experiments & Results](#experiments--results)
6. [API Key Setup](#api-key-setup)

---

## Introduction

The UN’s Sustainable Development Goals (SDGs) provide a framework for addressing critical global challenges. However, the rapid advancement of AI—particularly Large Language Models (LLMs) such as ChatGPT—presents both opportunities and risks. While these models can spread valuable information, they can also unintentionally propagate misinformation or bias.

**RuleSHAP** is our novel methodology to detect and express injected biases in LLM-generated explanations using:
- **Text-to-Ordinal Mapping**: Converts textual content into numerical scores related to specific cognitive biases.
- **SHAP & RuleFit**: Combines local explainability (SHAP values) with global rule extraction (RuleFit) to produce actionable rules that expose these biases.

This repository demonstrates the end-to-end process of:
1. Injecting biases into LLMs (e.g., GPT-4, LLaMA 3.1) via system instructions.
2. Using SHAP to analyse textual outputs.
3. Extracting global rules with **RuleSHAP** to detect and quantify biases.

---

## Abstract

The UN's Sustainable Development Goals (SDGs) serve as a blueprint for tackling global issues, while AI's advancement presents both opportunities and risks. Generative AI systems, like ChatGPT, can help spread information but also misinformation and biases, potentially undermining the SDGs. Detecting these biases is challenging for large language models (LLMs) usually operate with non-numerical inputs/outputs.

To address this, we show a bias detection methodology grounded in Explainable Artificial Intelligence (XAI). Our approach maps texts to numerical scores capturing cognitive biases linked to misinformation, enabling global XAI tools like SHAP and RuleFit to analyse LLM-generated content. We then examine the effects of deliberately injecting biases via system instructions in state-of-the-art LLMs, including ChatGPT-4 and LLaMA 3.1, revealing limitations in current XAI methods. Among these methods, SHAP comes close to detecting injected biases but cannot express them as actionable rules.

Hence, we introduce **RuleSHAP**, a new algorithm merging SHAP and RuleFit, increasing bias detection by 21% (MRR@1). This allows us to analyse LLMs on topics like climate action (SDG 13), well-being (SDG 3), and gender equality (SDG 5).

---

## Requirements & Installation

- **Operating System**: Tested on **macOS 15.3** (or similar)  
- **Python**: Requires **Python 3.9**

### Installation Steps

Run the provided setup script:
 ```bash
 chmod +x setup.sh
 ./setup.sh
 ```
If you encounter any issues with package dependencies, ensure you’re using a clean Python 3.9 environment (e.g., via `conda` or `venv`).

---

## Usage

1. **Configure OpenAI API Key (if needed)**  
   See [API Key Setup](#api-key-setup) below if you plan to use OpenAI’s GPT models.

2. **Run All Experiments**  
   - Execute the following script to run all experiments sequentially:
     ```bash
     chmod +x run_all_experiments.sh
     ./run_all_experiments.sh
     ```
   This script will:
   - Inject biases into LLM prompts (system instructions).
   - Generate LLM responses.
   - Perform text-to-ordinal mappings.
   - Calculate SHAP values.
   - Extract global rules via RuleFit (RuleSHAP).

3. **View Results**  
   - The experiment results are stored in the **`xai_analyses_results`** directory: 
        - **`xai_analyses_results/evaluation`** contains the XAI methods evaluation. 
        - **`xai_analyses_results/rules`** contains the rules extracted by RuleSHAP.
        - **`xai_analyses_results/summary_plot`** contains the global SHAP results.
        - **`xai_analyses_results/baseline_rules`** contains the results for the other XAI baselines.
   - Look for `.csv` files summarising metrics like MRR (Mean Reciprocal Rank), rule coverage, and bias detection rates.

---

## API Key Setup

If you plan to use OpenAI’s GPT-based models (e.g., `GPT-3.5`, `GPT-4`) in these experiments, you’ll need an API key.

1. **Sign Up / Log In**  
   - Go to [OpenAI’s platform](https://platform.openai.com/signup/) and log in to your account.

2. **Get Your API Key**  
   - Navigate to **API Keys** ([link](https://platform.openai.com/api-keys)).
   - Click **Create new secret key** and copy it.

3. **Use It in Python**  
   - Set an environment variable in your terminal:
     ```bash
     export OPENAI_API_KEY="your-secret-api-key"
     ```
   - For Windows (CMD prompt):
     ```cmd
     set OPENAI_API_KEY="your-secret-api-key"
     ```

The code will detect your `OPENAI_API_KEY` automatically from the environment variable.