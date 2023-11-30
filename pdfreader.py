import tabula
import pandas as pd

def extract_tables_from_pdf(pdf_path):
    # Use tabula to read tables from the PDF
    # tables = tabula.read_pdf(pdf_path, pages='6-16', multiple_tables=False, output_path="tab3.csv")
    tables = tabula.read_pdf(pdf_path, pages='6-16', multiple_tables=False, )


    # Process and print each table
    for i, table in enumerate(tables):
        print(f"Table {i + 1}:")
        print(table)
        print("\n" + "-" * 40 + "\n")


# Specify the path to your PDF file
pdf_file_path = 'tab3.pdf'

# Extract tables from the PDF
extract_tables_from_pdf(pdf_file_path)

