from jobspy import scrape_jobs
import pandas as pd
from IPython.display import display, HTML
# ['indeed', 'zip_recruiter', 'linkedin']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

#search_term = "software engineer",
#search_term = "data scientist",
#search_term = "product manager",
#search_term = "cyber security",
#search_term = "IT",

# Others: consultant, technical analyst, quant finance, manaegement (This study addresses more broad roles that most CS students usually will get in to)
# Don't do 'LinkedIn' as it can tend to block requests
#, 'zip_recruiter'
def scrape_jobs_from_site(path, res_wanted, job):
    jobs = scrape_jobs(
        site_name= ['indeed', 'glassdoor', 'zip_recruiter'],#['zip_recruiter', 'glassdoor', 'indeed'],
        location='USA',
        search_term=job,
        results_wanted=res_wanted,
        hours_old=100, # 24hr
        linkedin_fetch_description=True,
        country_indeed='USA'
    ) 

    jobs.to_excel(path)

# Swe Jobs
swe_path = './jobs/swe_jobs_12_19.xlsx'
swe_res = 1000
swe_term = "Software engineer"
#scrape_jobs_from_site(swe_path, swe_res, swe_term)

# Data science Jobs
ds_path = './jobs/ds_jobs_12_19.xlsx'
ds_res = 1000
ds_term = "Data scientist"
#scrape_jobs_from_site(ds_path, ds_res, ds_term)

# Cybersecurity Jobs
cs_path = './jobs/cs_jobs_12_19.xlsx'
cs_res = 1000
cs_term = "Cyber Security"
#scrape_jobs_from_site(cs_path, cs_res, cs_term)

# Product management Jobs
pm_path = './jobs/pm_jobs_12_19.xlsx'
pm_res = 1000
pm_term = "Technical product manager"
#scrape_jobs_from_site(pm_path, pm_res, pm_term)

# IT Jobs
it_path = './jobs/it_jobs_12_19.xlsx'
it_res = 1000
it_term = "IT"
#scrape_jobs_from_site(it_path, it_res, it_term)