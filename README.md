# Software Developer Salary Prediction

This Data Science/Machine Learning project uses the Stack Overflow Developer Survey 2024 data to train a ML model using three features to predict the salary.
The features used were "Country", "Education Level", and "Years in Professional Coding" to predict "Salary".
`sklearn` was used to find the best regressor and save the computed model to `model.pkl` file.

You can find the training process in `training.py`

Then, a streamlit app was made to make predictions and explore the data. (`app.py`, `predict_page.py`, `explore_page.py`)

I will try to upgrade  the project to use more features and/or display more data.
