# code-analysis-bcc-classification-results
This is the official repository for the code analysis of the paper "Improving Basal Cell Carcinoma Diagnosis with Real-Time AI Assistance for LC-OCT: a multi-centric retrospective study".


# Installation:

Retrive the code by cloning this repository

### Docker installation (linux)
For full reproducibility, this code comes with a docker environment that can be installed following those steps:
- go to `cd ./code-analysis-bcc-classification-results/`
- build the docker image and install all poetry dependencies using make `make install`
- launch jupyter by typing `make notebook`
- then follow the notebooks url to access the jupyter environment in which you can run the code 


### poetry environment

If you do not want to use the docker container, you can simply create a corresponding poetry environment using the `pyproject.toml` file.




# Data availibity

The `data` folder contains three csv files which contain all the results of our study.

**`data/simple_diags_CT.csv`**
This csv (";" separated) is a mapper from all the diagnosis to normalized categories
- 'case_id': unique identifier of the corresponding lesion
- 'histological_diagnosis': diagnosis as written by the pathologist
- 'category': normalized category associated to the histological diagnosis

**`data/BCC_Quiz_answers_anon.csv`**
This csv ("," separated) contains all the answer to each question by each doctor. The columns should be understood as follow:
- 'user': unique id of the doctor
- 'case_uuid': unique id of the the studied case
- 'is_diagnostic_bcc': histological result, True for BCC else False
- 'ai_assistance_present': was the AI assistance present for this question during the LC-OCT phase
- 'body_location': body location of the lesion
- 'clinical_phase_answer': answer to the "Is it BCC?" question at the clinical imagery phase
- 'clinical_phase_trust_score': confidence score given to the diagnosis at the clinical imagery phase
- 'clinical_phase_elapsed_time': time in seconds for answering at the clinical imagery phase
- 'clinical_phase_remark': free comment of the doctor on this specific case
- 'lcoct_phase_answer':  answer to the "Is it BCC?" question at the lc-oct imagery phase
- 'lcoct_phase_trust_score':  confidence score given to the diagnosis at the lc-oct imagery phase
- 'lcoct_phase_elapsed_time': time in seconds for answering at the lc-oct imagery phase
- 'lcoct_phase_remark': free comment of the doctor on this specific case at the lc-oct imagery phase
- 'answered_order': number indicating after how many previous questions this case was answered
- 'date': date at which the question was answered
- 'img_name': duplicate of case_uuid
- 'max_moving_avg_24': max moving average over 24 consecutive frames of the AI model on this LC-OCT images
- 'all_scores': frame level scores of the AI model
- 'user_type': level of expertise of the doctor

**`data/BCC_Quiz_answers_anon.csv`**
This csv file ("," separated) contains case level (not answer level) ai predictions scores.
- 'station_name': identifier of the device used by doctor when acquiring this case
- 'patient_id': unique identifier of the patient
- 'study_id': unique identifier of the exam number
- 'series_nb': number corresponding to this image in the study
- 'img_name': unique identifier of the image inside the exam number
- 'img_type': original img type (2D video or 3D)
- 'patient_sex': sex of the patient as written by the doctor during exam
- 'body part': body part as entered by the doctor during exam
- 'max_moving_average': maximum moving average score of 24 consecutive frames of the AI model
- 'all_scores': all scores of the AI model at the frame level
- 'case_uuid': duplicate of image_name
- 'is_diagnostic_bcc': histological result, True for BCC else False

# Code structure

The function used to compute all the statistic and make the plots of the study are located in the `src` folder.

- `src/CI_utils.py` contains function to compute sensitivity and specificity confidence intervals
- `src/plot_utils.py` contains function to make the plots of the publication
- `notebooks/Compute_statistics.ipynb` contains all the code to reproduce the statistical results of the paper.

# Time to run

The notebook should run in a few seconds (1 min max) on a modern laptop.