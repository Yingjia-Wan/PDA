import os
import json
from openai import OpenAI
from tqdm import tqdm  # Import tqdm for progress bars
import random
import re

# Set up OpenAI API environment
os.environ["OPENAI_API_KEY"] = "sk-BODTQM0S7LRU6lNJBf796fDaBf9c48319b0e55A6CeB1332f"
os.environ["OPENAI_BASE_URL"] = "https://open.xiaoai.one/v1"

client = OpenAI()

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

# Function to extract sections from gpt4o_nl
def extract_sections(gpt4o_nl):
    problem_pattern = re.compile(r"# Problem:(.*?)(?=# Explanation:|# Proof:|$)", re.DOTALL)
    explanation_pattern = re.compile(r"# Explanation:(.*?)(?=# Proof:|$)", re.DOTALL)
    proof_pattern = re.compile(r"# Proof:(.*?)(?=# Theorem:|$)", re.DOTALL)
    
    problem = problem_pattern.search(gpt4o_nl)
    explanation = explanation_pattern.search(gpt4o_nl)
    proof = proof_pattern.search(gpt4o_nl)
    
    return {
        "problem": problem.group(1).strip() if problem else "",
        "explanation": explanation.group(1).strip() if explanation else "",
        "proof": proof.group(1).strip() if proof else ""
    }

# Load prompt template
prompt_template = read_prompt_template('../prompt/fewshot_informalization.txt')
# Load random 10 samples from the random_test.json file
samples = read_samples('../data/random_test.json', num_samples=10, seed=42)

# Prepare output data list
new_samples = []

# Iterate over the samples, generate the new output, and collect results
# Wrap the loop with tqdm for the progress bar
for sample in tqdm(samples, desc="Processing samples"):
    # Replace {Theorem} in the template with the actual "output" value
    prompt_input = prompt_template.replace("{Theorem}", sample['output'])
    
    # Generate response from GPT-4o with temperature set to 0.7
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a math expert familiar with the Lean 4 theorem prover, a tool used for formal verification of mathematical theorems and proofs."},
            {"role": "user", "content": prompt_input}
        ],
        temperature=0.7
    )

    # Extract the response text
    gpt4o_output = completion.choices[0].message.content
    
    # Extract sections from gpt4o_nl
    sections = extract_sections(gpt4o_output)
    
    # Append to new samples
    new_samples.append({
        "nl": sample["input"],
        "formal": sample["output"],
        "prompt_input": prompt_input,
        "gpt4o_output": gpt4o_output,
        "nl_problem": sections["problem"],
        "nl_explanation": sections["explanation"],
        "nl_proof": sections["proof"]
    })

# Write the new output data to the random_test.json file
write_output('../FormL4/random_test_sample10.json', new_samples)

print("Processing complete! Results saved to ../output/random_test_sample10.json.")