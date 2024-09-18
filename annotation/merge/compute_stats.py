import json

def contains_t_or_f(annotation):
    """Helper function to determine if the annotation contains 'T' or 'F'."""
    if annotation:
        annotation = annotation.upper()  # Normalize to uppercase for safety
        if 'T' in annotation:
            return 'T'
        elif 'F' in annotation:
            return 'F'
    return None

def calculate_agreement_and_avg_t(subset, annotator_a, annotator_b, field_prefixes):
    """Calculate agreement and average T rate between two annotators."""
    agreement_counts = {field: 0 for field in field_prefixes}
    total_counts = {field: 0 for field in field_prefixes}
    t_counts = {field: 0 for field in field_prefixes}
    
    for entry in subset:
        for field in field_prefixes:
            # Get the annotations for the current field
            val_a = contains_t_or_f(entry.get(f'{field}_{annotator_a}'))
            val_b = contains_t_or_f(entry.get(f'{field}_{annotator_b}'))
            
            if val_a and val_b:
                # Check if the two annotations agree
                if val_a == val_b:
                    agreement_counts[field] += 1
                total_counts[field] += 1
                
                # Count occurrences of 'T'
                if val_a == 'T':
                    t_counts[field] += 1
                if val_b == 'T':
                    t_counts[field] += 1
    
    results = {}
    for field in field_prefixes:
        # Calculate agreement rate
        agreement_rate = agreement_counts[field] / total_counts[field] if total_counts[field] else 0
        
        # Calculate average T rate
        avg_t_rate = t_counts[field] / (2 * total_counts[field]) if total_counts[field] else 0
        
        results[field] = {
            'Agreement rate': agreement_rate * 100,
            'Average T rate': avg_t_rate * 100
        }
    
    return results

# Load the merged JSON file
models = ['gemini', 'gpt4o']
for model in models:
    print("#" * 10, model, "#" * 10)
    with open(f'{model}_sample10_merged_annotation.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split the data into two subsets
    subset_1 = data[:5]  # First 5 samples
    subset_2 = data[5:-1]  # Last 5 samples except last 1 #TODO
    
    # Fields to calculate stats for
    field_prefixes = ['informalization_success', 'informal_proof_correctness', 'model_preference']
    
    # Calculate agreement and average T for subset_1 (annotator 1 vs annotator 2)
    print("Calculating for first 5 samples (annotator 1 vs annotator 2)")
    results_1 = calculate_agreement_and_avg_t(subset_1, 1, 2, field_prefixes)
    
    # Calculate agreement and average T for subset_2 (annotator 3 vs annotator 2)
    print("Calculating for last 4 samples (annotator 3 vs annotator 2)")
    results_2 = calculate_agreement_and_avg_t(subset_2, 3, 2, field_prefixes)
    
    # Combine the results
    combined_results = {}
    for field in field_prefixes:
        combined_results[field] = {
            'Agreement rate': (results_1[field]['Agreement rate'] + results_2[field]['Agreement rate']) / 2,
            'Average T rate': (results_1[field]['Average T rate'] + results_2[field]['Average T rate']) / 2
        }
    
    # Print final combined results
    print("\nCombined results:")
    for field, stats in combined_results.items():
        print(f"{field}:")
        print(f"  Average T rate: {stats['Average T rate']:.2f}%")
        print(f"  Agreement rate: {stats['Agreement rate']:.2f}%")

