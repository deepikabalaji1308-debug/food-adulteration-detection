# Food Adulteration Detection Using Machine Learning

## Project Overview
Food adulteration is a major food-quality concern in which inferior or unauthorized substances may be added to food products. This project uses Machine Learning to analyze food adulteration data and predict the recorded severity level.

The application provides an interactive dashboard built using Streamlit and compares two Machine Learning algorithms: Random Forest and Gradient Boosting.

## Objectives
- Analyze food adulteration data.
- Apply Random Forest and Gradient Boosting algorithms.
- Compare model performance using evaluation metrics.
- Visualize model accuracy and confusion matrices.
- Predict the severity level for user-provided food product details.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## Machine Learning Algorithms
1. Random Forest Classifier
2. Gradient Boosting Classifier

## Project Features
- Interactive Streamlit dashboard
- Model performance comparison
- Confusion matrix visualization
- Food product information input
- Predicted severity level
- Model evaluation metrics

## Project Files
- `app.py` — Streamlit dashboard application
- `requirements.txt` — Required Python libraries
- `cleaned_food_adulteration_data.csv` — Dataset used by the application
- `train_models.py` — Script for training and comparing models, if included

## Installation and Execution
1. Clone or download this repository.
2. Install the required libraries:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

4. Open the local URL displayed in the terminal.

## Deployment
The application can be deployed using Streamlit Community Cloud. Once deployed, users can access the dashboard through a web browser.

## Important Note
The model predictions depend on the dataset and its recorded labels. If the dataset contains synthetic or randomized records, the evaluation results are for demonstration purposes only. This project is not a substitute for laboratory testing of physical food samples.

## Future Enhancements
- Use verified, real-world food adulteration data.
- Add more food categories and relevant features.
- Improve model performance and validation.
- Integrate laboratory sensor data for practical screening.

## Conclusion
This project demonstrates how Machine Learning algorithms can be used to analyze food adulteration data, compare predictive models, and present results through an interactive dashboard.
