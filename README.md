# Course-Job Fit Repository

## Overview
This repository contains the datasets, code, and analysis for the research described in "Course-Job Fit: Understanding the Contextual Relationship Between Computing Courses and Employment Opportunities" (Kverne et al., 2025).

## Repository Structure

### Datasets

**Job Descriptions:**
- Location: `Cleaned Datasets (100k jobs)`
- Contents: Organized collections of job descriptions across multiple computing domains:
  - Computer Science (CS)
  - Software Engineering (SWE)
  - Data Science (DS)
  - Information Technology (IT)
  - Project Management (PM)

**Course Information:**
- Location: 
  - `Cleaned Datasets` (Excel format for convenient analysis)
  - `all_courses`, `core_courses`, `elective_courses` (Original PDF syllabi)

### Code Modules

**Data Collection:**
- Location: `Fetch Jobs`
- Purpose: Scripts and utilities for automated job description collection

**Transformer Models:**
- Location: Various similarity modules:
  - `BGE_similarities`
  - `e5_similarities`
  - `SBERT_similarities`
  - `GTE_similarities`
  - `MPNet_similarities`
- Purpose: Text-to-embedding conversion utilities for semantic analysis

**Analysis:**
- Location: 
  - `Compare Models` (Course ranking analysis)
  - `analyze_top_bottom_courses` (Detailed course analysis including:)
    - Keyword extraction
    - Core vs. elective course comparison
    - Correlation with high-compensation employment opportunities

## Citation
When using this repository or referencing this work, please cite:
Christopher L. Kverne, Federico Monteverdi, Agoritsa Polyzou, Christine Lisetti, Janki Bhimani
“Course-Job Fit: Understanding the Contextual Relationship Between Computing Courses and Employment Opportunities”