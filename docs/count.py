from PyPDF2 import PdfFileReader
from datetime import datetime


today = datetime.now()
with open("main.pdf", "rb") as pdf_file:
    pdf_reader = PdfFileReader(pdf_file)
    print(today, pdf_reader.numPages)

