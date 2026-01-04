# Complaint Summarizer and Analyzer System (Using Logistic Regression)

## Project Description
This project implements a **Complaint Summarizer and Analyzer System** using basic Natural Language Processing (NLP) techniques and **Logistic Regression**.  
The system preprocesses textual complaints, builds a vocabulary, analyzes word importance, classifies complaints, and visualizes results using graphs.  
It is designed as a simple, educational lab project to demonstrate text preprocessing, feature extraction, classification, and analysis using Python.


## Project Members

- **Abdullah** — 22SP-005-CS  
- **Hashir Ul Wara** — 22SP-025-CS  
- **Eesha Tir Razia** — 22SP-061-CS  
- **Shajia Fatima** — 22SP-063-CS  


## Project Structure

```

Complaint-Summarizer-Analyzer/
│
├── src/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore

````

---

## Requirements
- Python 3.10 or above
- Required Python libraries are listed in `requirements.txt`

---

## Demo & Reproducibility Steps

Follow the steps below to run the project successfully:

### 1. Clone the Repository
```bash
git clone <https://github.com/etrazia17/Complaint_Summarizer-Analyzer_System>
cd Complaint-Summarizer-Analyzer
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Required NLP Resources

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Run the Project

```bash
python src/main.py
```

---

## Output

* Text preprocessing and complaint analysis in terminal
* Graphical visualization using Matplotlib

---

## Notes

* The source code has not been modified beyond standard execution requirements.
* All NLP resources are installed externally to maintain code integrity.




