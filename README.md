# Smart Complaint Management System Using NLP

## Overview

Smart Complaint Management System is a Python-based application that uses Natural Language Processing (NLP) to analyze complaints and help prioritize issues according to their urgency.

The application provides complaint classification, sentiment analysis, urgency detection, user authentication, and complaint tracking through a Streamlit interface.

## Key Features

- **Complaint Classification:** Processes complaint text for classification.
- **Sentiment Analysis:** Analyzes the sentiment expressed in complaint descriptions.
- **Urgency Detection:** Identifies potentially urgent complaints using text-based indicators.
- **User Authentication:** Supports user registration and login.
- **Complaint Tracking:** Maintains complaint records for tracking and review.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Text Processing | NLTK |
| Sentiment Analysis | TextBlob |
| Machine Learning | Scikit-learn |
| Data Handling | Pandas |

## System Workflow

1. The user submits a complaint through the application.
2. The complaint text is processed for analysis.
3. Text classification and sentiment analysis are applied.
4. Urgency indicators are evaluated to help prioritize complaints.
5. Complaint information is recorded for tracking and review.

## Project Structure

```text
smart-complaint-app/
├── app.py
├── train.py
├── complaints.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

The structure above reflects the files documented in the current project README. Update it if your actual repository contains a different set of files.

- `app.py` — Main Streamlit application.
- `train.py` — Model training script.
- `complaints.csv` — Complaint dataset, if this is the dataset used by your implementation.
- `model.pkl` — Serialized trained model.
- `vectorizer.pkl` — Serialized text vectorizer.
- `requirements.txt` — Python dependencies.

## Installation and Execution

### Prerequisites

- Python 3.10 or another version supported by the project's dependencies
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/prathyushabeerelli/smart-complaint-app.git
cd smart-complaint-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Ensure that the required model and vectorizer files are available and that their paths match those expected by the application.

## Model Evaluation

Model performance should be documented using appropriate evaluation metrics, such as accuracy, precision, recall, and F1-score, where applicable.

Evaluation results should be added after running the model on a suitable test dataset. No performance figures are reported here because verified results have not yet been provided.

## Potential Improvements

- Multilingual complaint processing.
- Integration of an AI-assisted response system.
- Voice-based complaint submission.
- Analytics dashboard for complaint trends.
- More comprehensive model evaluation.

These items are proposed improvements, not claims about currently implemented functionality.

## Author

**Prathyusha Beerelli**

B.Tech, Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
SR University

[GitHub Profile](https://github.com/prathyushabeerelli) | [LinkedIn](https://www.linkedin.com/in/prathyusha-beerelli-9a786131b/)
