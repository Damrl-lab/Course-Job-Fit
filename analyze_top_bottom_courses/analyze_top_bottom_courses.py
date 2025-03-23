import pandas as pd
from keybert import KeyBERT
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def get_courses(courses_df, course_name_series):
    # Convert series to list for matching
    course_names = course_name_series.tolist()
    
    # Filter courses that match any of the provided course names
    matched_courses = courses_df[courses_df['Course Name'].isin(course_names)].copy()

    # Sort the results to match the order of input course names
    matched_courses.loc[:, 'temp_sort'] = matched_courses['Course Name'].map({name: idx for idx, name in enumerate(course_names)})
    matched_courses = matched_courses.sort_values('temp_sort').drop('temp_sort', axis=1)
    
    return matched_courses


def topic_modeling(courses_df, min_topic_size=2):
    # Initialize BERTopic with custom parameters
    vectorizer_model = CountVectorizer(stop_words="english", 
                                     ngram_range=(1, 2),
                                     max_features=500)
    
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True
    )
    
    # Extract descriptions
    documents = courses_df['Course Description'].tolist()
    
    # Fit the model and transform the documents
    topics, probs = topic_model.fit_transform(documents)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Extract topics with their keywords and representative documents
    topic_analysis = {}
    for topic_id in topic_info['Topic'].unique():
        if topic_id != -1:  # Skip outlier topic
            topic_keywords = topic_model.get_topic(topic_id)
            topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]
            
            topic_analysis[topic_id] = {
                'keywords': [word for word, score in topic_keywords],
                'keyword_scores': [score for word, score in topic_keywords],
                'representative_docs': topic_docs,
                'doc_count': len(topic_docs)
            }
    
    return topic_analysis

# Returns topic keywords, and keywords more related to the teaching aspect, and noisy keywords
kw_model = KeyBERT() # Only initialize once
def extract_key_words(top_courses, bottom_courses=""):
    def process_descriptions(courses_df):
        # Single list to store all keywords from all courses
        all_keywords = []
        
        for _, row in courses_df.iterrows():
            description = row['Course Description']
            
            keywords = kw_model.extract_keywords(
                description,
                keyphrase_ngram_range=(1, 2),  # Allow for single words and bigrams
                stop_words='english',  # Remove common English stop words
                use_mmr=True,  # Use Maximal Marginal Relevance
                diversity=0.7,  # Diversity parameter
                top_n=15,  # Number of keywords to extract per course
            )
            all_keywords.extend(keywords)

        # Filter out less relevant key_words
        all_keywords = [(keyword, score) for keyword, score in all_keywords if score >= 0.3]
        return all_keywords
    
    # Process both top and bottom courses
    top_keywords = process_descriptions(top_courses)
    if bottom_courses != "" :
        bottom_keywords = process_descriptions(bottom_courses)
    bottom_keywords=""
    # Print results in a formatted way
    print("\nTop Courses Keywords:")
    print(top_keywords)

    #print("\nBottom Courses Keywords:")
    #print(bottom_keywords)
    
    return {
        'top_keywords': top_keywords,
        'bottom_keywords': bottom_keywords
    }

def analyze_courses(num_courses):
    courses_df = pd.read_excel('../Datasets/cleaned_all_courses.xlsx')

    programs = {
        'CS': '../compare_models/CS/course_rankings.xlsx',
        'DS': '../compare_models/DS/course_rankings.xlsx',
        'IT': '../compare_models/IT/course_rankings.xlsx',
        'PM': '../compare_models/PM/course_rankings.xlsx',
        'SWE': '../compare_models/SWE/course_rankings.xlsx'
    }

    high_paying_programs = {
        'CS': '../compare_models/CS/highest_paying_courses.xlsx',
        'DS': '../compare_models/DS/highest_paying_courses.xlsx',
        'IT': '../compare_models/IT/highest_paying_courses.xlsx',
        'PM': '../compare_models/PM/highest_paying_courses.xlsx',
        'SWE': '../compare_models/SWE/highest_paying_courses.xlsx'
    }
    
    for program, file_path in programs.items(): # programs.items():
        print(f"------------ Analyzing {program} topics/keywords ------------")
        program_df = pd.read_excel(file_path)['Course Name']
        
        # Get top courses
        top_courses = program_df[:num_courses]
        top_details = get_courses(courses_df, top_courses)
        #print(f"\nTop {num_courses} {program} courses:")
        #print(top_details[['Course Name', 'Course Description']])
        
        # Get bottom courses
        #bottom_courses = program_df[-num_courses:]
        #bottom_details = get_courses(courses_df, bottom_courses)
        #print(f"\nBottom {num_courses} {program} courses:")
        #print(bottom_details[['Course Name', 'Course Description']])

        # Perform topic modeling on all courses
        #print(f"\nTopic Analysis for {program} Top Courses:")
        #top_topics = topic_modeling(top_details)
        #for topic_id, topic_data in top_topics.items():
        #    print(f"\nTopic {topic_id}:")
        #    print(f"Keywords: {', '.join(topic_data['keywords'][:5])}")
        #    print(f"Number of courses: {topic_data['doc_count']}")
        
        #print(f"\nTopic Analysis for {program} Bottom Courses:")
        #bottom_topics = topic_modeling(bottom_details)
        #for topic_id, topic_data in bottom_topics.items():
            #print(f"\nTopic {topic_id}:")
            #print(f"Keywords: {', '.join(topic_data['keywords'][:5])}")
            #print(f"Number of courses: {topic_data['doc_count']}")


        extract_key_words(top_details, "")

analyze_courses(21)

"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import yake
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

courses = pd.read_excel('../Datasets/cleaned_all_courses.xlsx')
course_rankings = pd.read_excel('../compare_models/course_rankings.xlsx')

#print(top_course_names)
#print(bottom_course_names)

#print(top_courses)
#print(bottom_courses)

# Topic Modeling
# Keyword Extraction and Comparison
# Semantic Role Labeling
# Semantic Role Labeling

def perform_topic_modeling(texts, n_clusters=5):
    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embeddings for all texts
    embeddings = model.encode(texts)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Extract keywords for each cluster
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    
    cluster_keywords = []
    for i in range(n_clusters):
        cluster_texts = [text for j, text in enumerate(texts) if clusters[j] == i]
        if cluster_texts:
            try:
                X = vectorizer.fit_transform(cluster_texts)
                words = vectorizer.get_feature_names_out()
                cluster_keywords.append(words)
            except:
                cluster_keywords.append([])
    
    return clusters, cluster_keywords

def analyze_courses(top_courses, bottom_courses):
    # Combine descriptions for each group
    top_descriptions = top_courses['Course Description'].tolist()
    bottom_descriptions = bottom_courses['Course Description'].tolist()
    
    # Perform topic modeling on each group separately
    top_clusters, top_cluster_keywords = perform_topic_modeling(top_descriptions)
    bottom_clusters, bottom_cluster_keywords = perform_topic_modeling(bottom_descriptions)
    
    # Keyword Extraction with YAKE
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,  # ngram size
        dedupLim=0.7,
        top=20,
        features=None
    )
    
    def extract_keywords(texts):
        all_keywords = []
        for text in texts:
            keywords = kw_extractor.extract_keywords(text)
            all_keywords.extend([kw[0] for kw in keywords])
        return Counter(all_keywords)
    
    top_keywords = extract_keywords(top_descriptions)
    bottom_keywords = extract_keywords(bottom_descriptions)
    
    # Find distinctive keywords
    def get_distinctive_keywords(kw1, kw2, threshold=0.7):
        distinctive = {}
        all_words = set(list(kw1.keys()) + list(kw2.keys()))
        
        for word in all_words:
            count1 = kw1.get(word, 0)
            count2 = kw2.get(word, 0)
            if count1 + count2 > 0:
                ratio = count1 / (count1 + count2)
                if ratio > threshold:
                    distinctive[word] = ratio
        
        return distinctive
    
    top_distinctive = get_distinctive_keywords(top_keywords, bottom_keywords)
    bottom_distinctive = get_distinctive_keywords(bottom_keywords, top_keywords)
    
    return {
        'top_cluster_keywords': top_cluster_keywords,
        'bottom_cluster_keywords': bottom_cluster_keywords,
        'top_keywords': dict(top_keywords.most_common(20)),
        'bottom_keywords': dict(bottom_keywords.most_common(20)),
        'top_distinctive': dict(sorted(top_distinctive.items(), key=lambda x: x[1], reverse=True)[:20]),
        'bottom_distinctive': dict(sorted(bottom_distinctive.items(), key=lambda x: x[1], reverse=True)[:20])
    }

# Load your data
courses = pd.read_excel('../Datasets/cleaned_all_courses.xlsx')
course_rankings = pd.read_excel('../compare_models/aggregated_course_rankings.xlsx')

top_course_names = course_rankings[0:20]['Course Name']
bottom_course_names = course_rankings[len(course_rankings) - 20: len(course_rankings)]['Course Name']
top_courses = courses[courses['Course Name'].isin(top_course_names)]
bottom_courses = courses[courses['Course Name'].isin(bottom_course_names)]

# Run the analysis
results = analyze_courses(top_courses, bottom_courses)

# Print results
print("\nTop Course Topics:")
for i, keywords in enumerate(results['top_cluster_keywords']):
    print(f"Topic {i+1}: {', '.join(keywords)}")

print("\nBottom Course Topics:")
for i, keywords in enumerate(results['bottom_cluster_keywords']):
    print(f"Topic {i+1}: {', '.join(keywords)}")

print("\nTop Course Distinctive Keywords:")
for kw, score in results['top_distinctive'].items():
    print(f"{kw}: {score:.2f}")

print("\nBottom Course Distinctive Keywords:")
for kw, score in results['bottom_distinctive'].items():
    print(f"{kw}: {score:.2f}")

# Visualize keyword differences
plt.figure(figsize=(15, 8))
top_kw = list(results['top_distinctive'].keys())[:10]
top_scores = list(results['top_distinctive'].values())[:10]

plt.barh(top_kw, top_scores)
plt.title('Top 10 Most Distinctive Keywords in High-Ranked Courses')
plt.xlabel('Distinctiveness Score')
plt.tight_layout()
plt.show()
"""