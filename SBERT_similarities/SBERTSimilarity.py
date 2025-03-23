import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import time

# Model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
max_token_len = model.max_seq_length
tokenizer = model.tokenizer

# How many descriptions are above token limit
truncation_count = 0
total_count = 0

# Converts text into an embedding
def encode_text(text):
    global truncation_count, total_count
    with torch.no_grad():
        tokens = tokenizer(text, truncation=False)['input_ids'] # array of tokens
        if len(tokens) > max_token_len:
            truncation_count+=1
            embedding = get_mean_pooled_embedding(tokens)
        else:
            embedding = model.encode(text, convert_to_tensor=True)
        total_count+=1
        return embedding.cpu()

# When the token limit exceeds the models max capacity
def get_mean_pooled_embedding(tokens):
    chunks = [tokens[i:i + max_token_len] for i in range(0, len(tokens), max_token_len)]
    embeddings = []
    token_counts = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunk_embedding = model.encode(chunk_text, convert_to_tensor=True)
        embeddings.append(chunk_embedding)
        token_counts.append(len(chunk))

    total_tokens = sum(token_counts)
    weighted_embeddings = [embedding * (count / total_tokens) for embedding, count in zip(embeddings, token_counts)]
    final_embedding = torch.stack(weighted_embeddings).sum(dim=0)
    return final_embedding

def calculate_similarity(course_path, output_path, job_path):
    # Load cleaned course and job descriptions
    courses_df = pd.read_excel(course_path)
    jobs_df = pd.read_excel(job_path)

    # Extract the course descriptions and job descriptions
    course_names = courses_df['Course Name'].tolist()
    course_descriptions = courses_df['Course Description'].tolist()

    job_titles = jobs_df['title'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()
    job_salaries = jobs_df['mean_salary'].tolist()

    # Compute embeddings
    start_time = time.time()
    course_embeddings = [encode_text(desc) for desc in course_descriptions]
    job_embeddings = [encode_text(desc) for desc in job_descriptions]

    print(f"Elapsed time: {time.time() - start_time} seconds")  # Fixed
    print(f"Total descriptions truncated: {truncation_count}, tt_description : {total_count}")

    # Convert lists to tensors for similarity calculation
    course_embeddings = torch.stack(course_embeddings)
    job_embeddings = torch.stack(job_embeddings)

    # Compute cosine similarities between each course and job description
    similarity_matrix = util.cos_sim(course_embeddings, job_embeddings)

    # Create a list to store the results in the desired format
    results = [
        [course_name, job_title, similarity_matrix[i][j].item(), job_salaries[j]]
        for i, course_name in enumerate(course_names)
        for j, job_title in enumerate(job_titles)
    ]

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Course Name', 'Job Title', 'Similarity', 'Job Salary'])

    # Save the similarity results to an Excel file
    results_df.to_csv(output_path, index=False)
    print(f"Similarity between courses and jobs calculated and saved to '{output_path}'.")

course_path = '../Datasets/cleaned_all_courses.xlsx'
calculate_similarity(course_path, './SBERT_all_course_cs_jobs.csv', '../Datasets/cs_jobs.xlsx')
calculate_similarity(course_path, './SBERT_all_course_ds_jobs.csv', '../Datasets/ds_jobs.xlsx')
calculate_similarity(course_path, './SBERT_all_course_it_jobs.csv', '../Datasets/it_jobs.xlsx')
calculate_similarity(course_path, './SBERT_all_course_pm_jobs.csv', '../Datasets/pm_jobs.xlsx')
calculate_similarity(course_path, './SBERT_all_course_swe_jobs.csv', '../Datasets/swe_jobs.xlsx')
