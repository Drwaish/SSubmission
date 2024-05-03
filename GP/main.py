''' Diagnose disease base on prompt '''
import os
from torch import bfloat16
from transformers import pipeline
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
# from create_query import processor
from create_query import DataProcessor


load_dotenv()
# MODEL_NAME = "TheBloke/medalpaca-13B-GPTQ"
MODEL_NAME =  "HuggingFaceH4/zephyr-7b-beta"
data_processor = DataProcessor()

class QueryProcessor:
    def __init__(self, model_name = MODEL_NAME, model_directory_env='MODEL_DIRECTORY'):
        """
        Initializes the query processor with the necessary model and environment settings.

        Parameters
        ----------
        model_name
            The name of the model to use for processing queries.
         query_processor
            The external query processor object to format the query.
        model_directory_env
            Environment variable name for the model directory.
        """
        # load_dotenv()
        self.model_directory = os.getenv(model_directory_env)
        self.model_name = model_name
        # self.llm = self.load_model_endpoint()
        self.pipeline = self.load_pipeline()


    def load_model_endpoint(self):
        """
        Load model endpoint for inference.

        Paramaters
        ----------
        None

        Return
        ------
        LLM Object
        """
        # load_dotenv()
        llm = HuggingFaceEndpoint(
        repo_id=self.model_name,
        task="text-generation",
        max_new_tokens=1024,
        top_k= 5,
        temperature = 0.1,
        repetition_penalty = 1.03,
        huggingfacehub_api_token = self.hf_token  # Replace with your actual huggingface token
        ) 
        return llm
    def load_pipeline(self):
        """
        Loads the text generation pipeline using the specified model.

        Returns:
            A pipeline object configured for text generation.
        """

        return pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=bfloat16,
            device_map="auto",
        )
        
    def process_query(self, query : str , query_type : str):
        """
        Processes a query and generates a response using the loaded model.

        Parameters
        ----------
        query 
          The user query to be processed.

        query_type
            Query Processing Either related to lab or gp.
                gp means diagnose. 
                la means lab query. 

        Returns:
        --------
        str
            The generated response from the model.
        """
        # messages = self.query_processor.prepare_query_for_model(query=query)
        messages =  data_processor.prepare_query_for_model(query=query, query_type=query_type)

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95
        )
        # Return the last generated text assuming split by line break.
        return outputs[0]["generated_text"].strip().split("\n")[-1]
    def get_response(self, query : str = None , query_type : str = "gp"):
        """
        Generate Response base on query.

        Parameters
        ----------
        query
            Query on which this prediction will run.

        Return
        ------
        String    

        """
        response = self.process_query(query=query, query_type=query_type)
        # print(response)
        return response
        
if __name__ == "__main__":
    QUERY = """
    Lately, I've noticed reduced urine output, persistent fatigue, swelling in my ankles and legs,
        and frequent headaches. 
    """
    qp = QueryProcessor()
    print(qp.get_response(QUERY))
