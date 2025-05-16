# Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP

This repository contains the official code for the paper:

**“Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP”**

## Table of Contents

1. [Introduction](#introduction)
2. [Abstract](#abstract)
3. [Requirements & Installation](#requirements--installation)
4. [Usage](#usage)
5. [Experiments & Results](#experiments--results)
6. [API Key Setup](#api-key-setup)

---

## Brief Introduction

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

Generative AI systems can help spread information but also misinformation and biases, potentially undermining the UN Sustainable Development Goals (SDGs). Explainable AI (XAI) aims to reveal the inner workings of AI systems and expose misbehaviours or biases. However, current XAI tools, built for simpler models, struggle to handle the non-numerical nature of large language models (LLMs). This paper examines the effectiveness of global XAI methods, such as rule-extraction algorithms and SHAP, in detecting bias in LLMs. To do so, we first show a text-to-ordinal mapping strategy to convert non-numerical inputs/outputs into numerical features, enabling these tools to identify (some) misinformation-related biases in LLM-generated content. Then, we inject non-linear biases of varying complexity (univariate, conjunctive, and non-convex) into widespread LLMs like ChatGPT and Llama via system instructions, using global XAI methods to detect them. This way, we found that RuleFit struggles with conjunctive and non-convex biases, while SHAP can approximate conjunctive biases but cannot express them as actionable rules. Hence, we introduce RuleSHAP, a global rule extraction algorithm combining SHAP and RuleFit to detect more non-univariate biases, improving injected bias detection over RuleFit by +94% (MRR@1) on average.

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

To install Llama 3.1 using Ollama, follow these steps:

1. **Download and Install Ollama**

   - **For macOS and Windows:**
     - Visit the [Ollama website](https://ollama.com/download) and download the installer suitable for your operating system.
     - Run the installer and follow the on-screen instructions to complete the installation.

   - **For Linux:**
     - Open your terminal and execute the following command:
       ```bash
       curl -fsSL https://ollama.com/install.sh | sh
       ```
     - This command will download and install Ollama on your system.

2. **Install the Llama 3.1 Models**

   - Open your terminal (or Command Prompt on Windows).
   - Run the following commands to download and set up the Llama 3.1 models:
     ```bash
     ollama run llama3.1
     ```
    ```bash
     ollama run llama3.1:70b
     ```
   - The initial execution will download the model, which may take some time depending on your internet speed. Subsequent runs will use the locally stored model.

3. **Verify the Installation**

   - To ensure that Llama 3.1 is installed correctly, you can run a simple test:
     ```bash
     ollama run llama3.1 "Hello, Llama!"
     ```
   - If the installation is successful, the model will generate a response to the input prompt.

**Note:** Ensure your system meets the necessary hardware requirements for running Llama 3.1. For instance, the 8B model typically requires at least 32 GB of RAM and 8 GB of VRAM for optimal performance. ([github.com](https://github.com/kamalraj0611/llama-3-local-setup?utm_source=chatgpt.com))

For more detailed information and troubleshooting, refer to the [Ollama GitHub repository](https://github.com/ollama/ollama). 

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