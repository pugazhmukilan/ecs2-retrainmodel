import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

def retraing_model(combined_df: pd.DataFrame):
    global models  # Use the global variable to store the model
    
    # Ensure the DataFrame contains the expected columns
    combined_df.to_csv('datasetf.csv', index=False)
    expected_columns = ['moisture', 'temperature', 'humidity', 'action']
    if not all(col in combined_df.columns for col in expected_columns):
        raise ValueError(f"DataFrame must contain the columns: {expected_columns}")
    
    # Step 1: Data Preprocessing
    imputer = SimpleImputer(strategy='mean')
    combined_df[['moisture', 'temperature', 'humidity']] = imputer.fit_transform(combined_df[['moisture', 'temperature', 'humidity']])
    print("completed -- imputer")

    # Handle outliers (removing extreme outliers using IQR method)
    Q1 = combined_df[['moisture', 'temperature', 'humidity']].quantile(0.25)
    Q3 = combined_df[['moisture', 'temperature', 'humidity']].quantile(0.75)
    IQR = Q3 - Q1
    combined_df = combined_df[~((combined_df[['moisture', 'temperature', 'humidity']] < (Q1 - 1.5 * IQR)) | (combined_df[['moisture', 'temperature', 'humidity']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    print("completed -- outliers")

    # Handle multicollinearity using Variance Inflation Factor (VIF)
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(combined_df[['moisture', 'temperature', 'humidity']])
    print("completed -- standard scaler")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = ['moisture', 'temperature', 'humidity']
    vif_data["VIF"] = [variance_inflation_factor(scaled_df, i) for i in range(scaled_df.shape[1])]
    print("completed -- VIF")
    print(vif_data)

    # Step 2: Split data into features (X) and target (y)
    X = combined_df[['moisture', 'temperature', 'humidity']]
    y = combined_df['action']  # Target variable

    # Step 3: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Build the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("completed -- model training")
    
    # Step 5: Make predictions on the test set
    y_pred = model.predict(X_test)
    print("completed -- predicted")

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save the model
    joblib.dump(model, 'trained_model.pkl')
    models = model  # Update global variable

    return model


def predict(values):
    # Load the model
    model = joblib.load('trained_model.pkl')
    
    # Make a prediction
    prediction = model.predict([values])
    return prediction
