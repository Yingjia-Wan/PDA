import os
import json
from tqdm import tqdm
import random
import re

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason

# Set up Google Cloud Project
vertexai.init(project="bustling-surf-434708-h9", location="us-central1")  # Replace with your project ID

# Load the Gemini Pro model
model = GenerativeModel(model_name="gemini-1.5-pro-001")

# Function to read the prompt template from a file
def read_prompt_template(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to read data from a JSON file
def read_samples(file_path, num_samples=10, seed=42):
    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Randomly sample num_samples from the data
    sampled_data = random.sample(data, min(num_samples, len(data)))
    return sampled_data

# Function to write new samples to the output file
def write_output(file_path, new_data):
    with open(file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

# Function to extract sections from Gemini Pro output
def extract_sections(gemini_output):
    problem_pattern = re.compile(r"# Problem:(.*?)(?=# Explanation:|# Proof:|$)", re.DOTALL)
    explanation_pattern = re.compile(r"# Explanation:(.*?)(?=# Proof:|$)", re.DOTALL)
    proof_pattern = re.compile(r"# Proof:(.*?)(?=# Theorem:|$)", re.DOTALL)
    
    problem = problem_pattern.search(gemini_output)
    explanation = explanation_pattern.search(gemini_output)
    proof = proof_pattern.search(gemini_output)
    
    return {
        "problem": problem.group(1).strip() if problem else "",
        "explanation": explanation.group(1).strip() if explanation else "",
        "proof": proof.group(1).strip() if proof else ""
    }

# Load prompt template
prompt_template = read_prompt_template('../prompt/fewshot_informalization.txt')
# Load random 10 samples from the random_test.json file
samples = read_samples('../data/random_test.json', num_samples=10, seed=42)

# Generation parameters for Gemini Pro
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]

# Prepare output data list
new_samples = []

# Iterate over the samples, generate the new output, and collect results
for sample in tqdm(samples, desc="Processing samples"):
    # Replace {Theorem} in the template with the actual "output" value
    prompt_input = prompt_template.replace("{Theorem}", sample['output'])

    # Generate response from Gemini Pro
    response = model.generate_content(
        prompt_input,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # Extract the response text
    gemini_output = response.text

    # Extract sections from Gemini Pro output
    sections = extract_sections(gemini_output)
    
    # Append to new samples
    new_samples.append({
        "nl": sample["input"],
        "formal": sample["output"],
        "prompt_input": prompt_input,
        "gemini_output": gemini_output,
        "nl_problem": sections["problem"],
        "nl_explanation": sections["explanation"],
        "nl_proof": sections["proof"]
    })

# Write the new output data to the random_test_gemini.json file
write_output('../output/random_test_gemini_sample10.json', new_samples)

print("Processing complete! Results saved to ../output/random_test_gemini_sample10.json.")