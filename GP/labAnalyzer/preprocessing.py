import os
import requests
import shortuuid
import PyPDF2

class LabReport:
    def __init__(self):
        pass

    def download_report(self, url):
        """
        Download the report and save in  local labs folder.

        Parameters
        ----------
        url
        URL string for download data.
        
        Returns
        -------
        path of save lab
        """
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                file_name = str(shortuuid.ShortUUID().random(length=10))
                file_name = file_name+".pdf"
                file_path = os.path.join("labAnalyzer","labs", file_name)
                with open(file_path, "wb") as fil:
                    
                    fil.write(response.content)
                return file_path
            return response.status_code  # If request not successful then return response status
        except Exception as e:
            print("Exception while downloading lab report", e)

    def read_data(self, path: str):
        """
        Read data from pdf.

        Parameters
        ----------
        path
            Read data from pdf file.

        Return
        ------
        str
        """
        try:
            pdffileobj = open(path, 'rb')
            pdfreader = PyPDF2.PdfReader(pdffileobj)
            pageobj = pdfreader.pages[0]
            text = pageobj.extract_text().split("  ")
            report_query = text[0].replace("\n", " ")
            return report_query
        except Exception as e:
            print("Exception while reading lab report", e)

if __name__ == "__main__":
    URL = 'https://drive.google.com/uc?export=download&id=1eMXnTdKVGPPaNTdqJDrgptGf-xOfpoOY'
    lab = LabReport()
    path = lab.download_report(URL)
    if path:
        report_data = lab.read_data(path)
        print(report_data)
    
