# Sentiment Analysis of Women's E-commerce Clothing Reviews

This project demonstrates how sentiment analysis can be performed on e-commerce reviews using Python and various Natural Language Processing (NLP) techniques. The goal is to predict whether a customer review is positive (good) or negative (bad) based solely on the textual data provided.

## Dataset

We use the **Women's E-commerce Clothing Reviews** dataset from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).

## Overview

Sentiment analysis is an NLP technique that extracts emotions from raw text, commonly used on social media posts and customer reviews to determine if users are expressing positive or negative sentiments. In this project, we focus on customer reviews, predicting if a review is positive (good) or negative (bad) based on textual feedback.

### Problem Setup

Each review in the dataset contains:
- A textual feedback of the customer's experience.
- An overall rating, ranging from 1 to 5.

For simplicity, we classify the reviews into two categories:
- **Good reviews**: Ratings ≥ 3.
- **Bad reviews**: Ratings < 3.

The challenge is to predict this sentiment using only the raw text of the reviews.

## Approach

We use the following Python libraries for the analysis:
- **NLTK**: A popular module for NLP techniques.
- **Gensim**: A toolkit for topic modeling and vector space modeling.
- **Scikit-learn**: The most widely used machine learning library in Python.

### Steps

1. **Load the Dataset**: We first load the dataset and prepare it for analysis.
   
2. **Data Cleaning**:
   - Convert text to lowercase.
   - Tokenize the text (split into words) and remove punctuation.
   - Remove words containing numbers.
   - Remove stop words like "the", "a", "this", etc.

3. **Part-of-Speech (POS) Tagging**: 
   - Assign a tag to each word to define its grammatical category (noun, verb, etc.) using the WordNet lexical database.
   
4. **Lemmatization**: 
   - Transform words into their root forms (e.g., "rooms" → "room", "slept" → "sleep").

5. **Sentiment Analysis with VADER**: 
   - We use the **VADER** sentiment analysis tool to compute sentiment scores for each review, including:
     - **Negative (neg)**: Proportion of negative sentiment in the text.
     - **Neutral (neu)**: Proportion of neutral sentiment in the text.
     - **Positive (pos)**: Proportion of positive sentiment in the text.
     - **Compound**: A single score that sums up the sentiment.

6. **Feature Extraction**:
   - Add some simple metrics:
     - **Number of characters** in the text.
     - **Number of words** in the text.

7. **Word2Vec with Gensim**: 
   - We use **Word2Vec**, which creates numerical vector representations of each word in the corpus using shallow neural networks.

8. **TF-IDF (Term Frequency - Inverse Document Frequency)**: 
   - We compute TF-IDF values for every word and every document to evaluate the importance of words in the corpus.

9. **Modeling**:
   - Use **Random Forest Classifier** for prediction.
   - Train the model to classify reviews as "good" or "bad" based on the processed text data.

### Evaluation

For evaluating the model, we use the following metrics:
- **ROC Curve** (Receiver Operating Characteristic) to visualize the trade-off between sensitivity and specificity.
- **Average Precision (AP)** to measure precision-recall performance.

