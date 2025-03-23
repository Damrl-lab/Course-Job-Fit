import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Functions.CleanText import clean_text

"""
swe_paths = [
    './jobs/swe_jobs_10_05.xlsx',
    './jobs/swe_jobs_10_08.xlsx',
    './jobs/swe_jobs_10_10.xlsx',
    './jobs/swe_jobs_10_13.xlsx',
    './jobs/swe_jobs_10_16.xlsx',
    './jobs/swe_jobs_10_17.xlsx',
    './jobs/swe_jobs_10_24.xlsx',
    './jobs/swe_jobs_10_25.xlsx',
    './jobs/swe_jobs_10_28.xlsx',
    './jobs/swe_jobs_11_7.xlsx',
    './jobs/swe_jobs_11_13.xlsx',
    './jobs/swe_jobs_11_16.xlsx',
    './jobs/swe_jobs_11_19.xlsx',
    './jobs/swe_jobs_11_20.xlsx',
    './jobs/swe_jobs_11_21.xlsx',
    './jobs/swe_jobs_11_23.xlsx',
    './jobs/swe_jobs_11_27.xlsx',
    './jobs/swe_jobs_11_28.xlsx',
    './jobs/swe_jobs_11_29.xlsx',
    './jobs/swe_jobs_12_1.xlsx',
    './jobs/swe_jobs_12_2.xlsx',
    './jobs/swe_jobs_12_3.xlsx',
    './jobs/swe_jobs_12_4.xlsx',
    './jobs/swe_jobs_12_6.xlsx',
    './jobs/swe_jobs_12_8.xlsx',
    './jobs/swe_jobs_12_9.xlsx',
    './jobs/swe_jobs_12_11.xlsx',
    './jobs/swe_jobs_12_12.xlsx',
    './jobs/swe_jobs_12_13.xlsx',
    './jobs/swe_jobs_12_17.xlsx',
    './jobs/swe_jobs_12_19.xlsx',
]

cs_paths = [
    './jobs/cs_jobs_10_10.xlsx',
    './jobs/cs_jobs_10_13.xlsx',
    './jobs/cs_jobs_10_16.xlsx',
    './jobs/cs_jobs_10_17.xlsx',
    './jobs/cs_jobs_10_25.xlsx',
    './jobs/cs_jobs_10_28.xlsx',
    './jobs/cs_jobs_11_7.xlsx',
    './jobs/cs_jobs_11_13.xlsx',
    './jobs/cs_jobs_11_16.xlsx',
    './jobs/cs_jobs_11_19.xlsx',
    './jobs/cs_jobs_11_20.xlsx',
    './jobs/cs_jobs_11_21.xlsx',
    './jobs/cs_jobs_11_23.xlsx',
    './jobs/cs_jobs_11_27.xlsx',
    './jobs/cs_jobs_11_28.xlsx',
    './jobs/cs_jobs_11_29.xlsx',
    './jobs/cs_jobs_12_1.xlsx',
    './jobs/cs_jobs_12_2.xlsx',
    './jobs/cs_jobs_12_3.xlsx',
    './jobs/cs_jobs_12_6.xlsx',
    './jobs/cs_jobs_12_8.xlsx',
    './jobs/cs_jobs_12_9.xlsx',
    './jobs/cs_jobs_12_12.xlsx',
    './jobs/cs_jobs_12_13.xlsx',
    './jobs/cs_jobs_12_17.xlsx',
    './jobs/cs_jobs_12_19.xlsx',
]
ds_paths = [
    './jobs/ds_jobs_10_10.xlsx',
    './jobs/ds_jobs_10_13.xlsx',
    './jobs/ds_jobs_10_16.xlsx',
    './jobs/ds_jobs_10_17.xlsx',
    './jobs/ds_jobs_10_24.xlsx',
    './jobs/ds_jobs_10_28.xlsx',
    './jobs/ds_jobs_11_7.xlsx',
    './jobs/ds_jobs_11_13.xlsx',
    './jobs/ds_jobs_11_16.xlsx',
    './jobs/ds_jobs_11_19.xlsx',
    './jobs/ds_jobs_11_20.xlsx',
    './jobs/ds_jobs_11_21.xlsx',
    './jobs/ds_jobs_11_23.xlsx',
    './jobs/ds_jobs_11_27.xlsx',
    './jobs/ds_jobs_11_28.xlsx',
    './jobs/ds_jobs_11_29.xlsx',
    './jobs/ds_jobs_12_1.xlsx',
    './jobs/ds_jobs_12_2.xlsx',
    './jobs/ds_jobs_12_3.xlsx',
    './jobs/ds_jobs_12_4.xlsx',
    './jobs/ds_jobs_12_6.xlsx',
    './jobs/ds_jobs_12_8.xlsx',
    './jobs/ds_jobs_12_9.xlsx',
    './jobs/ds_jobs_12_11.xlsx',
    './jobs/ds_jobs_12_12.xlsx',
    './jobs/ds_jobs_12_13.xlsx',
    './jobs/ds_jobs_12_17.xlsx',
    './jobs/ds_jobs_12_19.xlsx',
]
it_paths = [
    './jobs/it_jobs_10_13.xlsx',
    './jobs/it_jobs_10_16.xlsx',
    './jobs/it_jobs_10_17.xlsx',
    './jobs/it_jobs_10_25.xlsx',
    './jobs/it_jobs_10_28.xlsx',
    './jobs/it_jobs_11_7.xlsx',
    './jobs/it_jobs_11_13.xlsx',
    './jobs/it_jobs_11_16.xlsx',
    './jobs/it_jobs_11_19.xlsx',
    './jobs/it_jobs_11_20.xlsx',
    './jobs/it_jobs_11_21.xlsx',
    './jobs/it_jobs_11_23.xlsx',
    './jobs/it_jobs_11_27.xlsx',
    './jobs/it_jobs_11_27.xlsx',
    './jobs/it_jobs_11_29.xlsx',
    './jobs/it_jobs_12_1.xlsx',
    './jobs/it_jobs_12_2.xlsx',
    './jobs/it_jobs_12_3.xlsx',
    './jobs/it_jobs_12_4.xlsx',
    './jobs/it_jobs_12_6.xlsx',
    './jobs/it_jobs_12_8.xlsx',
    './jobs/it_jobs_12_9.xlsx',
    './jobs/it_jobs_12_12.xlsx',
    './jobs/it_jobs_12_17.xlsx',
    './jobs/it_jobs_12_19.xlsx',
]
pm_paths = [
    './jobs/pm_jobs_10_10.xlsx',
    './jobs/pm_jobs_10_13.xlsx',
    './jobs/pm_jobs_10_17.xlsx',
    './jobs/pm_jobs_10_25.xlsx',
    './jobs/pm_jobs_10_28.xlsx',
    './jobs/pm_jobs_11_7.xlsx',
    './jobs/pm_jobs_11_13.xlsx',
    './jobs/pm_jobs_11_16.xlsx',
    './jobs/pm_jobs_11_19.xlsx',
    './jobs/pm_jobs_11_20.xlsx',
    './jobs/pm_jobs_11_21.xlsx',
    './jobs/pm_jobs_11_23.xlsx',
    './jobs/pm_jobs_11_27.xlsx',
    './jobs/pm_jobs_11_28.xlsx',
    './jobs/pm_jobs_11_29.xlsx',
    './jobs/pm_jobs_12_1.xlsx',
    './jobs/pm_jobs_12_2.xlsx',
    './jobs/pm_jobs_12_3.xlsx',
    './jobs/pm_jobs_12_6.xlsx',
    './jobs/pm_jobs_12_8.xlsx',
    './jobs/pm_jobs_12_9.xlsx',
    './jobs/pm_jobs_12_11.xlsx',
    './jobs/pm_jobs_12_12.xlsx',
    './jobs/pm_jobs_12_13.xlsx',
    './jobs/pm_jobs_12_17.xlsx',
    './jobs/pm_jobs_12_19.xlsx',
]
"""

def combine_dataframes(paths, field):
    tt_jobs = 0
    valid_jobs = 0
    all_dfs = []

    for path in paths:
        try:
            df = pd.read_excel(path)
            tt_jobs += df.shape[0]
            # Remove Empty Job Descriptions
            df.dropna(subset=['description'], inplace=True)
            valid_jobs += df.shape[0]
            # Add dataframe to combined dataframe
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")
            print("Skipping this file and continuing...")
            continue
    # Combine all dataframes to 1 large one and drop duplicates
    combined_df = pd.concat(all_dfs)
    combined_df.drop_duplicates(subset=['description'], inplace=True)

    # Results
    print(f'FIELD: {field} TT_JOBS: {tt_jobs} NON_EMPTY_JOBS: {valid_jobs} UNIQUE_VALID_JOBS: {combined_df.shape[0]}')

    return combined_df


# Load stopwords
def clean_data(df):
    df = df.reset_index(drop=True)

    # Convert salaries to numeric values
    df['min_amount'] = pd.to_numeric(df['min_amount'], errors='coerce')
    df['max_amount'] = pd.to_numeric(df['max_amount'], errors='coerce')

    # Convert hourly to yearly salaries (Multiply by 40 (hours per week) and 52 (weeks a year))
    df.loc[df['interval'] == 'hourly', 'min_amount'] = df['min_amount'] * 40 * 52
    df.loc[df['interval'] == 'hourly', 'max_amount'] = df['max_amount'] * 40 * 52

    # Get average salary (by looking at salary range)
    df['mean_salary'] = (df['min_amount'] + df['max_amount']) / 2

    # Drop rows where the job description is NaN
    df = df.dropna(subset=['description'])

    # Apply text cleaning function to job descriptions
    df['cleaned_description'] = df['description'].apply(clean_text)

    return df

cs_paths = [os.path.join('./jobs/cs_jobs', f) for f in os.listdir('./jobs/cs_jobs')]
ds_paths = [os.path.join('./jobs/ds_jobs', f) for f in os.listdir('./jobs/ds_jobs')]
it_paths = [os.path.join('./jobs/it_jobs', f) for f in os.listdir('./jobs/it_jobs')]
pm_paths = [os.path.join('./jobs/pm_jobs', f) for f in os.listdir('./jobs/pm_jobs')]
swe_paths = [os.path.join('./jobs/swe_jobs', f) for f in os.listdir('./jobs/swe_jobs')]

combined_swe_df = combine_dataframes(swe_paths, 'SWE')
combined_cs_df = combine_dataframes(cs_paths, 'CS')
combined_ds_df = combine_dataframes(ds_paths, 'DS')
combined_it_df = combine_dataframes(it_paths, 'IT')
combined_pm_df = combine_dataframes(pm_paths, 'PM')

#cleaned_swe_jobs = clean_data(combined_swe_df)
#cleaned_cs_jobs = clean_data(combined_cs_df)
#cleaned_ds_jobs = clean_data(combined_ds_df)
#cleaned_it_jobs = clean_data(combined_it_df)
#cleaned_pm_jobs = clean_data(combined_pm_df)

#print(f"Num Swe Jobs : {len(cleaned_swe_jobs)}")
#print(f"Num CS Jobs : {len(cleaned_cs_jobs)}")
#print(f"Num DS Jobs : {len(cleaned_ds_jobs)}")
#print(f"Num IT Jobs : {len(cleaned_it_jobs)}")
#print(f"Num PM Jobs : {len(cleaned_pm_jobs)}")

#cleaned_swe_jobs.to_excel('../Datasets/swe_jobs.xlsx')
#cleaned_cs_jobs.to_excel('../Datasets/cs_jobs.xlsx')
#cleaned_ds_jobs.to_excel('../Datasets/ds_jobs.xlsx')
#cleaned_it_jobs.to_excel('../Datasets/it_jobs.xlsx')
#cleaned_pm_jobs.to_excel('../Datasets/pm_jobs.xlsx')


"""
# Converts to 1 large dataset of specified distribution
percentage = len(cleaned_swe_jobs) / 100
num_swe_jobs = len(cleaned_swe_jobs)
num_cs_jobs = round(percentage * 5) # len(cleaned_cs_jobs)
num_ds_jobs = round(percentage * 10) # len(cleaned_ds_jobs)
num_it_jobs = round(percentage * 5) # len(cleaned_it_jobs)
num_pm_jobs = round(percentage * 5) # len(cleaned_pm_jobs)

df_swe =  cleaned_swe_jobs.sample(n=num_swe_jobs, random_state=42)
df_cs = cleaned_cs_jobs.sample(n=num_cs_jobs, random_state=42)
df_ds = cleaned_ds_jobs.sample(n=num_ds_jobs, random_state=42)
df_it = cleaned_it_jobs.sample(n=num_it_jobs, random_state=42)
df_pm = cleaned_pm_jobs.sample(n=num_pm_jobs, random_state=42)

df_final = pd.concat([df_swe, df_cs, df_ds, df_it, df_pm])
print("num of jobs combined no duplciates: ", len(df_final))
df_final.drop_duplicates(subset=['description'], inplace=True)
print("final dataset has length: ", len(df_final))
df_final.to_excel('../Datasets/final_jobs.xlsx', index=False)
"""