# biblioteca de funçoes
from PyPDF2 import PdfReader

def tratamento_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)

    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text