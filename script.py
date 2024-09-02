import os
import argparse
import shutil
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import PyPDF2


with open('modelOVR.pkl',"rb") as f:
    model = pickle.load(f)

with open('vectorizer.pkl',"rb") as v:
    vectorizer = pickle.load(v)

category_mapping = {'ACCOUNTANT': 0,
 'ADVOCATE': 1,
 'AGRICULTURE': 2,
 'APPAREL': 3,
 'ARTS': 4,
 'AUTOMOBILE': 5,
 'AVIATION': 6,
 'BANKING': 7,
 'BPO': 8,
 'BUSINESS-DEVELOPMENT': 9,
 'CHEF': 10,
 'CONSTRUCTION': 11,
 'CONSULTANT': 12,
 'DESIGNER': 13,
 'DIGITAL-MEDIA': 14,
 'ENGINEERING': 15,
 'FINANCE': 16,
 'FITNESS': 17,
 'HEALTHCARE': 18,
 'HR': 19,
 'INFORMATION-TECHNOLOGY': 20,
 'PUBLIC-RELATIONS': 21,
 'SALES': 22,
 'TEACHER': 23}

stemmer = PorterStemmer()
def preprocess(txt):
    # Remove non-english characters, punctuation,special characters, digits, continous underscores and extra whitespace
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub(r'[^\w\s]|_', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    txt = txt.lower()
    txt = word_tokenize(txt)
    txt = [w for w in txt if not w in stopwords.words('english')]
    txt = [stemmer.stem(w) for w in txt]
    return ' '.join(txt)


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text



def categorize_resumes(input_dir):
    output_dir = os.path.join(input_dir, 'categorized_resumes')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(input_dir, 'categorized_resumes.csv')

    with open(output_csv, 'w') as csv_file:
        csv_file.write('filename,category\n')

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_fullpath = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_fullpath)
            cleaned = preprocess(text)
            text_tfidf = vectorizer.transform([cleaned])
            predicted_category = int(model.predict(text_tfidf)[0])
            predicted_category = next(key for key, value in category_mapping.items() if value == predicted_category)
            output_category_dir = os.path.join(output_dir, predicted_category)
            os.makedirs(output_category_dir, exist_ok=True)
            shutil.move(pdf_fullpath, os.path.join(output_category_dir, filename))

            with open(output_csv, 'a') as csv_file:
                csv_file.write(f"{filename},{predicted_category}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Categorize Resumes')
    parser.add_argument('input_dir', type=str, help='Path to directory containing resumes')
    args = parser.parse_args()

    categorize_resumes(args.input_dir)

    print('Resumes categorized and CSV file generated.')