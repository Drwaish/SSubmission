import txtai

embedding = txtai.Embeddings("index")

# Prepare Data

def update_vectordb(disease_description : str, parent_disease:str , 
                    original_clinical_finding: str, specialist_doctor_doctor:str):
  """
  Add data in vector store.

  Parameters
  ---------

  """
  temp = f"""
    <DISEASE DESCRIPTION> \n {disease_description} \n
    <ROOT DISESEAS> {parent_disease}\n <Original_Clinical_Finding> {original_clinical_finding} \n
      <SPECIALIST DOCTOR> <<<{specialist_doctor_doctor}>>>> treat the diseses.

      """

                                
print("Enter the following thins : ")
