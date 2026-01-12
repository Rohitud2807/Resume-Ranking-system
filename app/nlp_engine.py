import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


if __name__ == "__main__":
    pdf_path = "data/sample_resumes/resume1.pdf"

    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    print("------ RAW TEXT ------")
    print(raw_text[:1000])   # print first 1000 chars

    print("\n------ CLEANED TEXT ------")
    print(cleaned_text[:1000])
