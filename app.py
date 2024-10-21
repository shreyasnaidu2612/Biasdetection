import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, precision_score, roc_curve, auc
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('dataset/new_lending_bias_dataset.csv')

# Prepare the dataset for training
def prepare_data(df):
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['gender', 'marital_status', 'race', 'loan_type'], drop_first=True)
    X = df_encoded.drop(columns=['loan_approved'])
    y = df_encoded['loan_approved']
    return X, y

X, y = prepare_data(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForest model with some regularization
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Evaluate model with cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=5)
print(f"Cross-validation scores: {cv_scores}")

# Get probabilities for the test set
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Calculate the best threshold based on ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
best_threshold_index = np.argmax(tpr - fpr)
best_threshold = thresholds[best_threshold_index]
print(f"Best threshold based on ROC: {best_threshold}")

# Business rule validation
def validate_loan(age, annual_income, credit_score, loan_amount, loan_duration, interest_rate):
    """Returns a message if the loan application fails business rules"""
    # Business rules
    if age < 18 or age > 70:
        return "Error: Age must be between 18 and 70."
    if loan_duration < 1 or loan_duration > 30:
        return "Error: Loan duration must be between 1 and 30 years."
    if annual_income < 20000:
        return "Error: Annual income must be at least $20,000."
    if loan_amount > 4 * annual_income:
        return "Error: Loan amount exceeds four times the annual income."
    if credit_score < 300 or credit_score > 850:
        return "Error: Credit score must be between 300 and 850."
    if credit_score < 600 and loan_amount > 0.3 * annual_income:
        return "Error: High loan amount for a low credit score."
    if interest_rate > 12:
        return "Error: Interest rate too high for approval."
    
    # If all business rules are satisfied, return None (no error)
    return None

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        age = int(request.form['age'])
        annual_income = int(request.form['annual_income'])
        credit_score = int(request.form['credit_score'])
        loan_amount = int(request.form['loan_amount'])
        loan_duration = int(request.form['loan_duration'])
        interest_rate = float(request.form['interest_rate'])
        
        # Business rule validation
        validation_error = validate_loan(age, annual_income, credit_score, loan_amount, loan_duration, interest_rate)
        if validation_error:
            # If validation fails, display the specific error message in a styled format
            return render_template('result.html', approval="Disapproved", reason=validation_error)

        # Extract other data from the form
        user_data = {
            'age': age,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'loan_duration': loan_duration,
            'interest_rate': interest_rate,
            'gender': request.form['gender'],
            'marital_status': request.form['marital_status'],
            'race': request.form['race'],
            'loan_type': request.form['loan_type']
        }
        
        # Create DataFrame for user input
        user_df = pd.DataFrame([user_data])
        
        # One-hot encoding the categorical variables
        user_encoded = pd.get_dummies(user_df, columns=['gender', 'marital_status', 'race', 'loan_type'], drop_first=True)
        
        # Align the user input with the training data columns
        user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)

        # Scale the user input
        user_scaled = scaler.transform(user_encoded)

        # Make prediction with dynamic threshold adjustment
        user_prob = model.predict_proba(user_scaled)[:, 1]
        loan_approval = (user_prob >= best_threshold).astype(int)[0]

        # If loan is disapproved based on model prediction
        if loan_approval == 0:
            return render_template('result.html', approval="Disapproved", reason="Your application did not meet the required criteria.")

        # Detect bias with AIF360
        binary_test = BinaryLabelDataset(df=X_test.assign(loan_approved=y_test), label_names=['loan_approved'], protected_attribute_names=['gender_Male'])
        metric_test = BinaryLabelDatasetMetric(binary_test, privileged_groups=[{'gender_Male': 1}], unprivileged_groups=[{'gender_Male': 0}])
        disparate_impact = metric_test.disparate_impact()

        # Send prediction and bias result to result.html
        return render_template('result.html', approval="Approved", disparate_impact=disparate_impact)

# Updated result.html template to handle disapproval

if __name__ == "__main__":
    app.run(debug=True)
