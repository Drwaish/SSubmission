import json
import txtai

class DataProcessor:
    def __init__(self, prompt_path: str = "sys_prompt.json", index_path: str = "index"):
        self.prompts = self.read_json(prompt_path)
        self.embeddings = txtai.Embeddings()
        self.embeddings.load(index_path)

    def read_json(self, path: str = "sys_prompt.json"):
        """
        Read JSON data from a file.

        Parameters
        ----------
        path
            Path to the JSON file.

        Returns
        -------
        dict
            Dictionary containing JSON data.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def retrieve_data(self, query: str):
        """
        Retrieve relevant data based on the query using embeddings.

        Parameters
        ----------
        query 
            Query string.

        Returns:
        --------
        list
            List of relevant data retrieved.
        """
        result = self.embeddings.search(query, 3)
        return result

    def prepare_prompt(self, query: str):
        """
        Prepare a prompt by combining query results with query string.

         Parameters
        ----------
        query
            Query string.

        Returns:
        -------
        str
            Prepared prompt string.
        """
        result = self.retrieve_data(query)
        data = ""
        for _, res in enumerate(result):
            temp = res['text']
            data = data + temp + "\n"
        data = "<<<<CONTEXT>>>> \n " + data + "\n <<<<QUERY>>>> \n" + query
        return data

    def prepare_query_for_model(self, query_type: str = "gp", query: str = "A infection in kidney"):
        """
        Prepare a query message for the model.

        Parameters
        ----------
        query_type
            Type of query (e.g., "gp" for general practitioner).
        query  
            Query string.

        Returns
        -------
        list
            List containing system and user messages.
        """
        gp_messages = [
            {"role": "system", "content": self.prompts['gp_prompt']},
            {"role": "user"}
        ]
        lab_messages = [
            {"role": "system", "content": self.prompts['lab_analysis']},
            {"role": "user"}
        ]

        if query_type == "gp":
            data = self.prepare_prompt(query=query)
            gp_messages[1]['content'] = data
            return gp_messages
        elif query_type == "la":
            lab_messages[1]['content'] = query
            return lab_messages

if __name__ == "__main__":
    processor = DataProcessor(index_path="index")
    query = """
        Lately, I've noticed reduced urine output, persistent fatigue, swelling in my ankles and legs,
        and frequent headaches. Additionally, I'm dealing with nausea and a loss of appetite, and my
        urine appears darker and cloudy."
        """
    gp_query = processor.prepare_query_for_model(query_type="gp", query=query)
    print(gp_query)
