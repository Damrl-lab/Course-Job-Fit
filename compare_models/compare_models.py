import pandas as pd
import itertools
import numpy as np

core_courses = ['CDA_3102', 'CEN_4010', 'CGS_1920', 'CGS_3095', 'CIS_3950', 'CIS_4951', 'CNT_4713', 'COP_2210', 'COP_3337', 'COP_3530', 'COP_4338', 'COP_4555', 'COP_4610', 'COT_3100', 'ENC_3249', 'MAD_2104']
elective_courses = ['CAP_4052', 'CAP_4104', 'CAP_4453', 'CAP_4506', 'CAP_4612', 'CAP_4630', 'CAP_4641', 'CAP_4710', 'CAP_4770', 'CAP_4830', 'CDA_4625', 'CEN_4021', 'CEN_4072', 'CEN_4083', 'CIS_4203', 'CIS_4731', 'COP_4226', 'COP_4520', 'COP_4534', 'COP_4604', 'COP_4655', 'COP_4710', 'COP_4751', 'COT_3510', 'COT_3541', 'COT_4431', 'COT_4521', 'COT_4601', 'CTS_4408', 'MAD_3301', 'MAD_3401', 'MAD_3512', 'MAD_4203', 'MHF_4302']

def compare_models(field):
    # Load similarity data
    sbert_sim = pd.read_csv(f'../computed_similarities/SBERT_NEW/SBERT_all_course_{field.lower()}_jobs.csv')
    mpnet_sim = pd.read_csv(f'../computed_similarities/MPnet_NEW/MPnet_all_course_{field.lower()}_jobs.csv')
    bge_sim = pd.read_csv(f'../computed_similarities/BGE_NEW/BGE_all_course_{field.lower()}_jobs.csv')
    gte_sim = pd.read_csv(f'../computed_similarities/GTE_NEW/GTE_all_course_{field.lower()}_jobs.csv')
    e5_sim = pd.read_csv(f'../computed_similarities/E5_NEW/e5_all_course_{field.lower()}_jobs.csv')

    # Function to compute average similarity rank
    def rank_courses_avg_similarity(df):
        avg_sims = df.groupby('Course Name')['Similarity'].mean().sort_values()
        return avg_sims

    # Compute average similarity ranks for each model
    models = {
        "MPNet": rank_courses_avg_similarity(mpnet_sim),
        "E5": rank_courses_avg_similarity(e5_sim),
        "GTE": rank_courses_avg_similarity(gte_sim),
        "SBERT": rank_courses_avg_similarity(sbert_sim),
        "BGE": rank_courses_avg_similarity(bge_sim)
    }

    # Helper function to rank courses for average similarity
    def get_avg_sim_ranks(avg_sims, model_name):
        avg_sims_rank = avg_sims.rank(ascending=False).astype(int)
        return pd.DataFrame({
            'Course Name': avg_sims.index,
            f'{model_name}_Avg_Sim_Rank': avg_sims_rank
        }).set_index('Course Name')

    # Create a DataFrame for all ranks
    all_ranks = None
    for model_name, avg_sims in models.items():
        ranks_df = get_avg_sim_ranks(avg_sims, model_name)
        if all_ranks is None:
            all_ranks = ranks_df
        else:
            all_ranks = all_ranks.join(ranks_df, how='outer')

    # Calculate pairwise rank differences
    def calculate_rank_differences(all_ranks):
        model_columns = all_ranks.columns
        model_pairs = list(itertools.combinations(model_columns, 2))

        rank_differences = {}
        for model1, model2 in model_pairs:
            diff = (all_ranks[model1] - all_ranks[model2]).abs()
            avg_diff = diff.mean()
            rank_differences[(model1, model2)] = avg_diff

        return rank_differences

    rank_differences = calculate_rank_differences(all_ranks)

    # Convert rank differences to a DataFrame
    rank_diff_df = pd.DataFrame([
        {"Model Pair": f"{pair[0]} vs {pair[1]}", "Average Rank Difference": diff}
        for pair, diff in rank_differences.items()
    ])

    # Calculate average similarity of each model to others
    def calculate_model_avg_similarity(rank_differences):
        model_sums = {}
        model_counts = {}

        for (model1, model2), avg_diff in rank_differences.items():
            model_sums[model1] = model_sums.get(model1, 0) + avg_diff
            model_sums[model2] = model_sums.get(model2, 0) + avg_diff
            model_counts[model1] = model_counts.get(model1, 0) + 1
            model_counts[model2] = model_counts.get(model2, 0) + 1

        model_avg_similarity = {model: model_sums[model] / model_counts[model] for model in model_sums}
        return model_avg_similarity

    model_avg_similarity = calculate_model_avg_similarity(rank_differences)

    # Convert model average similarity to DataFrame
    model_avg_similarity_df = pd.DataFrame({
        'Model': model_avg_similarity.keys(),
        'Average Similarity to Others': model_avg_similarity.values()
    }).sort_values(by='Average Similarity to Others')

    # Calculate Spearman correlations between models
    spearman_corr = all_ranks.corr(method='spearman')

    # Rank courses based on average similarity rank across models
    all_ranks['Average_Course_Rank'] = all_ranks.mean(axis=1)
    course_rankings = all_ranks.sort_values(by='Average_Course_Rank')

    # Save all results to Excel
    rank_diff_df.to_excel(f'./{field.upper()}/model_rank_differences.xlsx', index=False)
    model_avg_similarity_df.to_excel(f'./{field.upper()}/model_similarity_rank.xlsx', index=False)
    course_rankings.to_excel(f'./{field.upper()}/course_rankings.xlsx')
    spearman_corr.to_excel(f'./{field.upper()}/model_spearman_correlations.xlsx')

    # Print summary of results
    print("Most Similar Models:")
    print(rank_diff_df.loc[rank_diff_df["Average Rank Difference"].idxmin()])

    print("\nLeast Similar Models:")
    print(rank_diff_df.loc[rank_diff_df["Average Rank Difference"].idxmax()])

    print("\nModel Average Similarity to Others:")
    print(model_avg_similarity_df)

    print("\nTop Ranked Courses Based on Average Rank:")
    print(course_rankings.head())

#compare_models('cs')
#compare_models('ds')
#compare_models('it')
#compare_models('pm')
#compare_models('swe')

# Initialize empty lists for each model
def compare_ranges():  
    sbert_df = []
    mpnet_df = []
    bge_df = []
    gte_df = []
    e5_df = []

    # Process each field
    fields = ["CS", "DS", "IT", "PM", "SWE"]

    for field in fields:
        # Read files for each model and append to respective lists
        sbert_sim = pd.read_csv(f'../computed_similarities/SBERT_NEW/SBERT_all_course_{field.lower()}_jobs.csv')
        sbert_sim['field'] = field  # Add field column
        sbert_df.append(sbert_sim)

        mpnet_sim = pd.read_csv(f'../computed_similarities/MPnet_NEW/MPnet_all_course_{field.lower()}_jobs.csv')
        mpnet_sim['field'] = field
        mpnet_df.append(mpnet_sim)

        bge_sim = pd.read_csv(f'../computed_similarities/BGE_NEW/BGE_all_course_{field.lower()}_jobs.csv')
        bge_sim['field'] = field
        bge_df.append(bge_sim)

        gte_sim = pd.read_csv(f'../computed_similarities/GTE_NEW/GTE_all_course_{field.lower()}_jobs.csv')
        gte_sim['field'] = field
        gte_df.append(gte_sim)

        e5_sim = pd.read_csv(f'../computed_similarities/E5_NEW/e5_all_course_{field.lower()}_jobs.csv')
        e5_sim['field'] = field
        e5_df.append(e5_sim)

    # Combine dataframes for each model
    sbert_combined = pd.concat(sbert_df, ignore_index=True)
    mpnet_combined = pd.concat(mpnet_df, ignore_index=True)
    bge_combined = pd.concat(bge_df, ignore_index=True)
    gte_combined = pd.concat(gte_df, ignore_index=True)
    e5_combined = pd.concat(e5_df, ignore_index=True)

    model_dfs = {
        'SBERT': sbert_combined,
        'MPNet': mpnet_combined,
        'BGE': bge_combined,
        'GTE': gte_combined,
        'E5': e5_combined
    }

    for model_name, df in model_dfs.items():
        # Find the similarity column (assuming it's named 'Similarity')
        sim_col = 'Similarity'
        
        # Filter out negative values for minimum calculation
        min_sim = df[df[sim_col] >= 0][sim_col].min()
        max_sim = df[sim_col].max()
        avg_sim = df[sim_col].mean()
        
        print(f"\n{model_name} Model Statistics:")
        print(f"Minimum Similarity (excluding negatives): {min_sim:.4f}")
        print(f"Maximum Similarity: {max_sim:.4f}")
        print(f"Average Similarity: {avg_sim:.4f}")
        print("-" * 50)
        
        # Field-wise statistics with non-negative minimums
        print("Field-wise Statistics:")
        field_stats = df[df[sim_col] >= 0].groupby('field')[sim_col].agg(['min', 'max', 'mean'])
        print(field_stats.round(4))
        print("=" * 50)

#compare_ranges()

def top_paying_jobs(field):
    # Function to compute average similarity for top 100 paying jobs
    def rank_courses_avg_similarity(df, n): 
        # Get top 1% paying jobs
        top_jobs = df.sort_values('Job Salary', ascending=False).head(n)['Job Title'].unique()
        
        # Filter similarity data for top paying jobs
        top_jobs_data = df[df['Job Title'].isin(top_jobs)]
        
        # Calculate average similarity per course for top paying jobs
        avg_sims = top_jobs_data.groupby('Course Name')['Similarity'].mean().sort_values(ascending=False)
        
        return avg_sims

    # Load similarity data
    sbert_sim = pd.read_csv(f'../computed_similarities/SBERT_NEW/SBERT_all_course_{field.lower()}_jobs.csv')
    mpnet_sim = pd.read_csv(f'../computed_similarities/MPnet_NEW/MPnet_all_course_{field.lower()}_jobs.csv')
    bge_sim = pd.read_csv(f'../computed_similarities/BGE_NEW/BGE_all_course_{field.lower()}_jobs.csv')
    gte_sim = pd.read_csv(f'../computed_similarities/GTE_NEW/GTE_all_course_{field.lower()}_jobs.csv')
    e5_sim = pd.read_csv(f'../computed_similarities/E5_NEW/e5_all_course_{field.lower()}_jobs.csv')
    
    jobs_df = pd.read_excel(f'../Datasets/{field.lower()}_jobs.xlsx').dropna(subset=['mean_salary'])
    n = int(len(jobs_df) / 100)
    print(f'field {field} has {n} top paying')
    models = {
        "MPNet": rank_courses_avg_similarity(mpnet_sim, n),
        "E5": rank_courses_avg_similarity(e5_sim, n),
        "GTE": rank_courses_avg_similarity(gte_sim, n),
        "SBERT": rank_courses_avg_similarity(sbert_sim, n),
        "BGE": rank_courses_avg_similarity(bge_sim, n)
    }

    # Create a dictionary to store rankings for each model
    rankings = {}
    
    # Calculate rankings for each model
    for model_name, avg_sims in models.items():
        # Convert to rankings (1-based ranking)
        rankings[model_name] = pd.Series(np.arange(1, len(avg_sims) + 1), 
                                       index=avg_sims.index)
        
        #print(f"\nResults for {model_name} model ({field.upper()} field):")
        #print("\nTop 10 courses by average similarity:")
        #print(avg_sims.head(10))
        
    # Create a DataFrame with all rankings
    all_rankings = pd.DataFrame(rankings)
    
    # Calculate average ranking across all models
    all_rankings['Average Rank'] = all_rankings.mean(axis=1)
    
    # Sort by average ranking
    final_rankings = all_rankings.sort_values('Average Rank')
    
    print("\nFinal normalized rankings across all models:")
    print(f"\nTop 10 courses by average rank, FIELD: {field.upper()}:")
    print(final_rankings.head(10))
    final_rankings.to_excel(f'./{field.upper()}/highest_paying_courses.xlsx')
    
    return final_rankings

#top_paying_jobs('cs')
#top_paying_jobs('ds')
#top_paying_jobs('it')
#top_paying_jobs('pm')
#top_paying_jobs('swe')