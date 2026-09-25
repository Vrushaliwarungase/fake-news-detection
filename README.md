# Fake News Detection using Machine Learning

## Project Overview

This project is a Machine Learning based Fake News Detection system.
It classifies a news article as Fake News or Real News based on patterns
learned from a labeled news dataset.

The project uses Natural Language Processing (NLP) and Machine Learning
to convert text into numerical features and classify the news.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Machine Learning Workflow

1. Load the news dataset
2. Add labels to the data
3. Remove missing text values
4. Split the data into training and testing sets
5. Convert text into numerical features using TF-IDF
6. Train a Logistic Regression classifier
7. Evaluate the model
8. Save the trained model
9. Create a Streamlit web application for prediction

## Model Performance

The model achieved an accuracy of **98.63%** on the test split used in
this project.

## Application

The Streamlit application allows the user to enter a news article and
receive the model's classification.

## Important Note

The prediction represents the classification learned from the training
dataset. It should not be treated as factual verification of a news
article.

## Future Improvements

- Use a larger and more diverse dataset
- Try other Machine Learning algorithms
- Add precision, recall and F1-score
- Improve the user interface
- Add more robust fact-checking methods