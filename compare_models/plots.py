import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Read the data
CS_spearman = pd.read_excel('./CS/model_spearman_correlations.xlsx')
DS_spearman = pd.read_excel('./DS/model_spearman_correlations.xlsx')
IT_spearman = pd.read_excel('./IT/model_spearman_correlations.xlsx')
PM_spearman = pd.read_excel('./PM/model_spearman_correlations.xlsx')
SWE_spearman = pd.read_excel('./SWE/model_spearman_correlations.xlsx')

def clean_model_names(name):
    """Clean model names by removing suffixes"""
    name = name.replace('_Avg_Sim_Rank', '')
    return name

def average_correlation_matrices(dataframes):
    # Clean model names in all dataframes
    processed_dfs = []
    for df in dataframes:
        df_copy = df.copy()
        
        # Clean column names
        df_copy.columns = [clean_model_names(col) if col != 'Unnamed: 0' else col for col in df_copy.columns]
        
        # Clean index/row names in the first column
        df_copy['Unnamed: 0'] = df_copy['Unnamed: 0'].apply(clean_model_names)
        
        # Convert numeric columns to float
        numeric_columns = df_copy.columns[1:]
        df_copy[numeric_columns] = df_copy[numeric_columns].astype(float)
        
        processed_dfs.append(df_copy)
    
    # Initialize sum matrix with zeros
    sum_matrix = pd.DataFrame(0, 
                            index=processed_dfs[0].index, 
                            columns=processed_dfs[0].columns)
    
    # Sum all matrices
    for df in processed_dfs:
        sum_matrix.iloc[:, 1:] += df.iloc[:, 1:]
    
    # Calculate average
    avg_matrix = sum_matrix.copy()
    avg_matrix.iloc[:, 1:] = sum_matrix.iloc[:, 1:] / len(processed_dfs)
    
    # Keep the model names
    avg_matrix['Unnamed: 0'] = processed_dfs[0]['Unnamed: 0']
    
    return avg_matrix

def plot_correlation_heatmap(correlation_matrix):
    # Create a copy of the matrix without the 'Unnamed: 0' column
    plot_matrix = correlation_matrix.set_index('Unnamed: 0')
    
    # Set up the matplotlib figure with a smaller size
    plt.figure(figsize=(4, 3.5), dpi=300)  # Reduced figsize for smaller boxes
    
    # Create a black and white colormap
    colors = ['#053061', '#FFFFFF', '#67001F'] 
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Create heatmap with improved styling and no colorbar
    sns.heatmap(plot_matrix, 
                annot=True,
                cmap=cmap,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.2f',
                annot_kws={'size': 6},  # Adjusted annotation size
                cbar=False)
    
    # Remove axis labels
    plt.xlabel('')
    plt.ylabel('')
    
    # Make tick labels smaller
    plt.xticks(rotation=0, ha='center', fontsize=6)  # Reduced fontsize for tick labels
    plt.yticks(rotation=0, fontsize=6)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()

# Create list of all correlation matrices
correlation_matrices = [CS_spearman, DS_spearman, IT_spearman, PM_spearman, SWE_spearman]

# Calculate average correlations
average_correlations = average_correlation_matrices(correlation_matrices)

# Display the result with cleaner formatting
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print("\nAveraged Correlation Matrix:")
print(average_correlations)

# Create the heatmap
plot_correlation_heatmap(average_correlations)