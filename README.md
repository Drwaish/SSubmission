# Heal Labs 
## Empowering your Health Choices


A complete end-to-end holistic medical platform for patient and the users.

Complete Code available in **GP** folder. 

##Non-Commercial Usage
All rights are reserved by the publishers.

### Patient Usage  
- Pre-Diagnose based on symptoms enter by the user.
- Interact with chatbot regarding  you disease.
- Record Keeping and Medical Report Analyzer.

### Doctor Usage
- Provide Second opinion, segment the image,  to the user based on nifti image enter by doctor.

# How you can Setup the Repo

## Quick Easy Way

Execute project ProjectSetup.ipynb's notebbok.

**Alternatively**

### Create Virtual Environment
```
conda create -n heallabs
conda activatt heallabs
```
Now, clone the repo.
```
git clone https://github.com/Drwaish/SSubmission
cd SSubmission
```
### Install Requirements
Install requirements  using requirements.txt
```
pip install -r requirements.txt
```
Now 
```
cd GP
python app.py
```
Gradio App will be open.

### Environment Variable
Create **.env** file in GP contain Hugging face Token
```
HF_TOKEN=<Change with your HF_token>
```


# Medical Imaging
Medical images , specifcally brain CT scan , are segmented that helps doctor to make well informed decision. 

Kindly visit the **Imaging** folder.

**Recommendation**

It is recommended to use  GPU for  inference. 


