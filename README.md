**Repository Name:** Movie-Review-Sentiment-Analysis

**Description:**  
This project builds a system to classify movie reviews as positive or negative using NLP and machine learning, focusing on assisting movie enthusiasts. The model is trained on IMDB reviews, aiming for an F1 score of 0.85+, leveraging technologies like TF-IDF, Logistic Regression, and LightGBM.

---

# Movie Review Sentiment Analysis

## Overview
The **Movie Review Sentiment Analysis** project aims to build a model that can classify movie reviews from IMDB as either positive or negative. This project is part of an initiative by **Film Junky Union**, a community dedicated to movie aficionados who want to engage with classic films. By employing Natural Language Processing (NLP) techniques, we train a model that can automatically categorize reviews, making it easier for community members to quickly assess the sentiment of content related to their favorite films.

This project uses Python libraries for data preprocessing and modeling, and leverages a mix of traditional machine learning techniques, such as Logistic Regression, as well as more modern approaches, including LightGBM and pre-trained Transformer models, to reach an F1 score target of 0.85.

## Table of Contents
- [Objectives](#objectives)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Objectives
The main goals of this project are to:
1. Develop a machine learning model that can classify movie reviews as positive or negative.
2. Achieve an F1 score of at least 0.85 to ensure a high-quality categorization system.
3. Assist movie enthusiasts in quickly filtering and engaging with meaningful reviews.

## Technologies
- Python
- Scikit-learn
- Pandas & NumPy
- TF-IDF Vectorizer
- Logistic Regression
- LightGBM
- Transformers (Hugging Face)
- NLTK & SpaCy for text preprocessing

## Dataset
The dataset used for this project consists of IMDB movie reviews that are labeled as either **positive** or **negative**. This dataset is publicly available and frequently used for sentiment analysis tasks. It includes:
- Text of the movie reviews.
- Labels indicating the sentiment of the review (positive or negative).

Ensure the dataset is available in the correct path before running the scripts for training and evaluation.

## Model Pipeline
The sentiment analysis model follows these steps:
1. **Text Preprocessing:**
   - Removal of punctuation, stopwords, and numbers.
   - Lemmatization of words to reduce inflectional forms.
   - Tokenization using NLTK and SpaCy.

2. **Feature Extraction:**
   - TF-IDF Vectorization is used to transform the text data into numerical features suitable for machine learning models.

3. **Modeling:**
   - Logistic Regression and LightGBM are used to classify the reviews.
   - Transformers from Hugging Face are used for more nuanced understanding of sentiment when required.

4. **Evaluation:**
   - Models are evaluated based on precision, recall, and F1 score, with the target F1 score set at 0.85.

## Installation
To get started, you need to install the necessary dependencies. Run the following command to create the environment and install required libraries:

```sh
pip install -r requirements.txt
```

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/yourusername/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis
```

## Usage
To run the project, you can use the Jupyter notebook provided (`proyecto_14_clean.ipynb`) or execute the training script directly:
1. Ensure the dataset path is correctly set in the notebook or script.
2. Run the cells sequentially in the notebook to preprocess the data, train the model, and evaluate its performance.

For predictions on new reviews, you can use a separate script (`predict_sentiment.py`) which takes review text as input and outputs the sentiment prediction.

## Results
The model's performance is evaluated based on F1 score, precision, and recall. By leveraging both traditional machine learning models like Logistic Regression and advanced models like LightGBM and Transformers, the project achieved a robust F1 score exceeding 0.85, making it effective in detecting sentiment accurately.

## Contributing
Contributions to improve the model or expand its functionality are welcome! Please fork the repository and create a pull request with your enhancements. Make sure that your code is well-documented and tested.

## License
This project is licensed under the MIT License. You are free to use and modify it as per your needs.

---

### Author
- **Your Name** - [GitHub Profile](https://github.com/yourusername)

Feel free to reach out for questions, suggestions, or collaborations.
