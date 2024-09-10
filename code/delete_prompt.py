import json

# Function to read the original JSON file
def read_original_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to write the modified JSON file
def write_modified_json(file_path, modified_data):
    with open(file_path, 'w') as file:
        json.dump(modified_data, file, indent=4)

# Path to the original JSON file
original_json_file = '../output/random_test_gemini_sample10.json'
# Path to the new modified JSON file
modified_json_file = '../output/gemini_sample10.json'

# Read the original JSON data
original_data = read_original_json(original_json_file)

# Prepare the modified data list
modified_data = []

# Iterate over the original data and remove the 'prompt_input' field
for sample in original_data:
    # Remove the 'prompt_input' field
    sample.pop('prompt_input', None)
    # Append the modified sample to the modified data list
    modified_data.append(sample)

# Write the modified data to the new JSON file
write_modified_json(modified_json_file, modified_data)

print(f"Processing complete! Modified results saved to {modified_json_file}.")