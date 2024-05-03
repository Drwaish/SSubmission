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

# How you can Setup
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

**Recommendation**

It is recommended to use 24 Gb , GPU for fast inference. 


