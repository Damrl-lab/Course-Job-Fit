# course-job-fit

To Access Complete List of Cours syllabi, please visit https://www4.cis.fiu.edu/courses/Syllabi/
Data set of Jobs can be found at kaggle.com/100K+ American Tech Job Postings (2024 Oct-Dec)

This code includes the xxx things

1) Job Fetching and Data cleaning logic can be found in the fetch_jobs folder
2) Syllabi reading logic (reading the PDF files) can be found in the read_syllibi.py file in the main directory
3) Embedding logic and similarity calculation for each model can be found within their respective folder (for example SBERT_similarities or e5_similarities)
4) Course Ranking logic can be found it the rank_courses.py file within the main directory (it ranks courses based on 4 metrics using the results of each model)
5) Comparing the model result logic can be found in the comparedModelSimilarities.py file in the main directory.

Overall the code and structure of the project should be quite easy to follow. Moreover we encourage you to use the dataset we collected for your own research projects and compare the results with what we found!

Cite our paper:
Christopher L. Kverne, Federico Monteverdi, Agoritsa Polyzou, Christine Lisetti, Janki Bhimani
“Course-Job Fit: Understanding the Contextual Relationship Between CS Courses and Employment
Opportunities”