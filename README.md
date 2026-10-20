# Smart Complaint Management System

A Python-based complaint management application that uses Natural Language Processing (NLP) techniques to analyze complaint text and help identify issues requiring attention.

**Live Application:** [Smart Complaint Management System](https://college-complaints.streamlit.app/)  
**Source Code:** [GitHub Repository](https://github.com/prathyushabeerelli/smart-complaint-app)

## 1. Project Overview

The system provides a web interface for submitting and managing complaints. It applies text analysis techniques to support complaint processing, sentiment analysis, and urgency identification.

The project demonstrates the application of NLP and machine learning concepts in a practical complaint management workflow.

## 2. Key Features

- Complaint submission and management
- Text processing and classification
- Sentiment analysis
- Urgency identification using text-based indicators
- User registration and authentication
- Complaint record management

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Text Processing | NLTK |
| Sentiment Analysis | TextBlob |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |

## 4. Application Workflow

1. The user submits a complaint through the web interface.
2. The application processes the complaint text.
3. Text analysis techniques are applied to the complaint.
4. Urgency indicators help identify potentially critical issues.
5. Complaint records are maintained for subsequent review.

## 5. Installation and Setup

### Prerequisites

- Python installed on your system
- pip package manager
- Git

### Clone the repository

```bash
git clone https://github.com/prathyushabeerelli/smart-complaint-app.git
cd smart-complaint-app
```

### Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

Ensure that all required datasets, configuration files, and trained model files are available before running the application.

## 6. Model Evaluation

Model performance should be evaluated using an appropriate test dataset and relevant metrics, including precision, recall, F1-score, and accuracy where applicable.

Quantitative results should be added after testing the implemented model.

## 7. Potential Enhancements

- Multilingual complaint processing
- Voice-based complaint submission
- Complaint trend analysis and reporting
- Integration of AI-assisted response suggestions
- Expanded model evaluation and validation

These are potential future enhancements and are not necessarily part of the current implementation.

## 8. Author

**Prathyusha Beerelli**  
B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
SR University

**GitHub:** [prathyushabeerelli](https://github.com/prathyushabeerelli)  
**LinkedIn:** [Professional Profile](https://www.linkedin.com/in/prathyusha-beerelli-9a786131b/)

