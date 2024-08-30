####  **Chosen Model and Rationale**
Multiple machine learning models were trained and evaluated to determine the best approach for classifying resumes into categories. The models considered included:

- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest (RF)**
- **Decision Tree (DT)**

Each model was trained on the same dataset and evaluated using accuracy scores on both the training and testing sets. Among these, the **Random Forest (RF)** model was chosen as the final model due to its superior accuracy.

#### **Preprocessing and Feature Extraction**
For Preprocessing and feature extraction the following methods were applied:

- **Text Preprocessing**: 
  - **Tokenization**: The text was split into individual words (tokens).
  - **Stopword Removal**: Common words (e.g., "and," "the") that do not contribute much to the classification were removed.
  - **Stemming**: Words were reduced to their root form using the Porter Stemmer.
  - **Unnessecary character removal**: Non-english characters, punctuation,special characters, digits, continous underscores and extra whitespace were removed.

- **Feature Extraction**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: This technique was employed to convert textual data into numerical features. It captures the importance of words in the documents relative to their frequency across the corpus, which helps in emphasizing significant words while diminishing the influence of common but less important words.

The TF-IDF matrix served as the input features for the models, and the categories provided the labels.

## Setup Instructions

- **Clone the repo**
    
    ```bash
    git clone https://github.com/remon-rakibul/resume-categorization-ml.git
    ```

- **Install Required Libraries**

    Open a terminal or command prompt and navigate to the directory where you placed the files. Install the necessary libraries using the following command:

    ```bash
    pip install -r requirements.txt
    ```

- **Run the Script**
    Execute the script with the following command:
    ```bash
    python script.py path/to/dir
    ```
    Replace `path/to/dir` with the path to the folder containing your resume PDF files.

- **Expected Output**
    * The script will organize the PDF files into subdirectories based on their predicted categories.
    * A CSV file named `categorized_resumes.csv` will be generated in the `path/to/dir` folder. This file will contain two columns: `filename` and `category`.

    The `categorized_resumes.csv` file will have the following format:
    ```csv
    filename,category
    30563572.pdf,HR
    19717385.pdf,ARTS
    44476983.pdf,HR
    ```
    In this example, `30563572.pdf` and `44476983.pdf` were categorized as `HR`, while `19717385.pdf` was categorized as `ARTS`.