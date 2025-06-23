import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Constants
NUM_CUSTOMERS = 50000
TODAY = datetime(2025, 6, 1)

# Feature generators
def generate_age():
    return np.random.choice(range(18, 70), p=np.linspace(0.01, 1, 52)/sum(np.linspace(0.01, 1, 52)))

def generate_income(age):
    if age < 25:
        return np.random.randint(10000, 30000)
    elif age < 35:
        return np.random.randint(30000, 70000)
    elif age < 50:
        return np.random.randint(70000, 150000)
    else:
        return np.random.randint(30000, 80000)

def generate_transaction_data():
    freq = np.random.poisson(10)
    avg_value = np.random.randint(500, 5000)
    return freq, avg_value

def generate_digital_score(age):
    if age < 30:
        return np.random.randint(70, 100)
    elif age < 50:
        return np.random.randint(40, 90)
    else:
        return np.random.randint(10, 60)

def assign_segment(age, income, freq, investment, loans, digital_score):
    if income > 100000 and investment >= 2:
        return "High Net-Worth"
    elif income > 50000 and digital_score > 70:
        return "Mass Affluent"
    elif freq > 15 and loans > 0:
        return "SME Owner"
    elif income < 30000 and age < 25:
        return "Student"
    elif age > 60:
        return "Senior"
    else:
        return "Mass Market"

# Data generation
data = []

for i in range(NUM_CUSTOMERS):
    customer_id = f"CUST{i+1:05d}"
    age = generate_age()
    gender = np.random.choice(['Male', 'Female'])
    marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], p=[0.5, 0.4, 0.08, 0.02])
    income = generate_income(age)
    location = np.random.choice(['Urban', 'Semi-Urban', 'Rural'], p=[0.5, 0.3, 0.2])
    account_type = np.random.choice(['Savings', 'Current', 'Salary'])
    total_txn, avg_txn_value = generate_transaction_data()
    last_txn_days_ago = np.random.randint(1, 365)
    last_transaction_date = TODAY - timedelta(days=last_txn_days_ago)
    digital_score = generate_digital_score(age)
    loan_products = np.random.poisson(0.5)
    investment_products = np.random.poisson(1.0)

    recency = last_txn_days_ago
    monetary = total_txn * avg_txn_value

    segment = assign_segment(age, income, total_txn, investment_products, loan_products, digital_score)

    data.append([
        customer_id, age, gender, marital_status, income, location, account_type,
        total_txn, avg_txn_value, last_transaction_date.strftime("%Y-%m-%d"),
        recency, monetary, digital_score, loan_products, investment_products, segment
    ])

# Create DataFrame
columns = [
    'customer_id', 'age', 'gender', 'marital_status', 'income', 'location', 'account_type',
    'total_transactions', 'average_transaction_value', 'last_transaction_date',
    'recency', 'monetary', 'digital_channel_usage_score', 'loan_products', 'investment_products',
    'customer_segment'
]

df = pd.DataFrame(data, columns=columns)

# Preview
print(df.head())

# Optional: Save

df.to_excel(r"C:\Users\Bramarambika\Downloads\Workoopolis\Customer_segmentation\customer_segmentation_banking.xlsx", index=False)
