import pandas as pd

# --- 1. Define the Model Metrics ---

pvcnn_values = {
    'lgan_mmd-CD': 474.1660461425781,
    'lgan_mmd_smp-EMD': 2732.073974609375,
    'lgan_mmd_smp-CD': 335.8780517578125,
    'lgan_mmd-EMD': 3005.349365234375,
    'lgan_cov-EMD': 0.15625,
    'lgan_cov-CD': 0.3125,
    '1-NN-CD-acc_t': 0.375,
    '1-NN-CD-acc_f': 0.96875,
    '1-NN-CD-acc': 0.671875,
    '1-NN-EMD-acc_t': 0.96875,
    '1-NN-EMD-acc': 0.984375,
    'JSD': 0.026629936536614274
}

calopodit_values = {
    'lgan_mmd-CD': 363.5998229980469,
    'lgan_mmd_smp-EMD': 2680.424560546875,
    'lgan_mmd_smp-CD': 326.1539611816406,
    'lgan_mmd-EMD': 2740.2236328125,
    'lgan_cov-EMD': 0.1875,
    'lgan_cov-CD': 0.34375,
    '1-NN-CD-acc_t': 0.59375,
    '1-NN-CD-acc_f': 0.875,
    '1-NN-CD-acc': 0.734375,
    '1-NN-EMD-acc_t': 0.96875,
    '1-NN-EMD-acc': 0.984375,
    'JSD': 0.10879843043929771
}

# Define the optimal direction for sorting/highlighting
OPTIMAL_DIRECTION = {
    'lgan_mmd-CD': 'Lower', 'lgan_mmd_smp-EMD': 'Lower',
    'lgan_mmd_smp-CD': 'Lower', 'lgan_mmd-EMD': 'Lower',
    'lgan_cov-EMD': 'Higher', 'lgan_cov-CD': 'Higher',
    '1-NN-CD-acc_t': 'Lower', '1-NN-CD-acc_f': 'Lower',
    '1-NN-CD-acc': 'Lower', '1-NN-EMD-acc_t': 'Lower',
    '1-NN-EMD-acc': 'Lower', 'JSD': 'Lower',
}

# --- 2. Create and Populate DataFrame ---

# Combine the metrics into a dictionary for the DataFrame
data = {
    'PVCNN': pvcnn_values,
    'CALOPODIT': calopodit_values,
    'Optimal Direction': OPTIMAL_DIRECTION,
}

# Create DataFrame
df = pd.DataFrame(data)

# --- 3. Determine the Winner ---

def determine_winner(row):
    """Determines which model has the optimal score based on direction."""
    pvcnn_val = row['PVCNN']
    calopodit_val = row['CALOPODIT']
    direction = row['Optimal Direction']
    
    # Check for ties first
    if pvcnn_val == calopodit_val:
        return 'Tie'
        
    if direction == 'Lower':
        # Lower score is better
        return 'PVCNN' if pvcnn_val < calopodit_val else 'CALOPODIT'
    
    elif direction == 'Higher':
        # Higher score is better
        return 'PVCNN' if pvcnn_val > calopodit_val else 'CALOPODIT'
        
    return 'N/A' # Should not happen

# Apply the function to create the new Comparison column
df['Winner'] = df.apply(determine_winner, axis=1)

# Reorder columns for presentation
df = df[['PVCNN', 'CALOPODIT', 'Optimal Direction', 'Winner']]

# --- 4. Print the Final Table ---

print("--- Point Cloud Model Metric Comparison ---")
print(df.to_string(float_format="{:.4f}".format))