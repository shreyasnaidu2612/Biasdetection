import pandas as pd
import numpy as np

# Simulating a cleaner, more balanced dataset for bias detection in lending practices
np.random.seed(42)

# Simulate demographic data
n_samples = 1000
age = np.random.randint(18, 70, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
marital_status = np.random.choice(['Single', 'Married'], n_samples)
race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples)

# Simulate financial data
annual_income = np.random.randint(20000, 120000, n_samples)
credit_score = np.random.randint(300, 850, n_samples)
loan_amount = np.random.randint(5000, 50000, n_samples)
loan_duration = np.random.randint(1, 30, n_samples)  # Duration in years
loan_type = np.random.choice(['Personal Loan', 'Home Loan', 'Car Loan', 'Education Loan'], n_samples)
interest_rate = np.random.uniform(3, 15, n_samples)

# Simulate loan outcome and approval decision
loan_approved = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])  # More balanced approval rate
loan_defaulted = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% default rate

# Create a DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'marital_status': marital_status,
    'race': race,
    'annual_income': annual_income,
    'credit_score': credit_score,
    'loan_amount': loan_amount,
    'loan_duration': loan_duration,
    'loan_type': loan_type,
    'interest_rate': interest_rate,
    'loan_approved': loan_approved,
    'loan_defaulted': loan_defaulted
})

# Save the dataset to a CSV file
df.to_csv('dataset/new_lending_bias_dataset.csv', index=False)
