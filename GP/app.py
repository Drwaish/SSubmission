'''Gradio base User Interface Application for better input'''
import os
import gradio as gr
from disease_llm import create_llm_chain
from main import QueryProcessor
from labAnalyzer.preprocessing import LabReport

# Initialize your objects and functions
qp = QueryProcessor()
lr = LabReport()
chain = create_llm_chain()

def pre_diagnose(query):
    """
    Gradio user interface for better experience to diagnose the patient.

    Parameter
    ---------
    query
        Symptoms enter by the user.

    Return
    ------
    str
    """
    return qp.get_response(query)  # Ensure this is the correct function name

def process_lab_report(url  : str = None):
    """
    Gradio user interface for better experience to know about report.

    Parameter
    ---------
    url
        Load data from  url
    Return
    ------
    str
    """
    if url :
        file_path = lr.download_report(url=url)
    print("file_path", file_path)
    report_data = lr.read_data(file_path)

    return qp.get_response(query=report_data, query_type="la")

def chatbot(message, history):

    response = chain.run(message)
    # history.append((message, response))
    return response



with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.HTML("<h1><center> Heal Labs </center></h1>")
    gr.HTML("<h3><center> Empowering your Health Choices </center></h3>")

    tab1, tab2, tab3 = gr.Tab("Pre-Diagnose"), gr.Tab("Chatbot"), gr.Tab("Report Analyzer")
    with tab1:
        gr.HTML("<h2><center> Pre-Diagnosis </center></h2>")

        with gr.Row():
            with gr.Column():
                symptom_query = gr.TextArea(label="Enter Your Symptoms Here:")
            with gr.Column():
                diagnosis_output = gr.TextArea(label="Remedies", lines=4)
        btn = gr.Button("Generate")
        btn.click(pre_diagnose, inputs=[symptom_query], outputs=[diagnosis_output])

    with tab2:
        gr.HTML("<h2><center> ChatBot </center></h2>")
        gr.HTML("<h3><center> Interact with Chatbot </center></h3>")
        gr.ChatInterface(chatbot)
    with tab3:
        gr.HTML("<h2><center> Medical Report Analyzer </center></h2>")
        gr.HTML("<h3><center> Pathological Report Only </center></h3>")

        link = gr.Textbox(label="Paste report link here")
        diagnosis_output = gr.TextArea(label="Reports Insigts", lines=4)
        process_lab_report_btn = gr.Button("Process Report")
        process_lab_report_btn.click(process_lab_report, inputs=[link], outputs=[diagnosis_output])


demo.launch(share = True,  debug = True)
