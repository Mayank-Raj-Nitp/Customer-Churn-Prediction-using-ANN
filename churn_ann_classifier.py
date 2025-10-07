import pandas as pd
import numpy as np
import io, requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- DATA LOADING AND INITIAL CLEANING  ---

# Using a publicly available, clean-ish version of the Telco Churn dataset
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
try:
    s=requests.get(URL).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    print(f"Successfully loaded dataset with {len(df)} rows.")
except Exception as e:
    print(f"Error loading external data: {e}. Using small synthetic fallback.")
    # Fallback to keep the code runnable if the external link fails
    df = pd.DataFrame({
        'gender': np.random.choice(['Female', 'Male'], 100), 'SeniorCitizen': np.random.randint(0, 2, 100),
        'tenure': np.random.randint(1, 72, 100), 'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(20, 8000, 100), 'Partner': np.random.choice(['Yes', 'No'], 100),
        'Churn': np.random.choice(['Yes', 'No'], 100)
    })

# Converting 'TotalCharges' to numeric (it contains some ' ' strings which will be NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Droped customer ID and handle NaNs by dropping rows with missing values (Quick Cleaning)
df.drop('customerID', axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
# Encode the target variable (Churn: No=0, Yes=1)
le = LabelEncoder()
y = le.fit_transform(df['Churn'])

# --- DATA PREPARATION & FEATURE ENGINEERING ---

# Defining column types based on the Telco Churn dataset
# Numerical features that need scaling
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Categorical features that need One-Hot Encoding
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Creating Preprocessing Pipeline for features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Using ColumnTransformer to apply different transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though none are expected here
)

X_processed = preprocessor.fit_transform(X)
input_dim = X_processed.shape[1]

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# ---   ANN MODEL DEVELOPMENT AND TRAINING ---

# Building the Sequential Model with a Dropout layer for regularization
model = Sequential([
    # Input layer and first hidden layer
    Dense(units=64, activation='relu', input_dim=input_dim),
    Dropout(0.2), # Regularization to prevent overfitting
    # Second hidden layer
    Dense(units=32, activation='relu'),
    # Output layer (sigmoid for binary classification)
    Dense(units=1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
print("\n--- Training the Customer Churn ANN (Real Data) ---")
model.fit(
    X_train, y_train,
    epochs=10,          # Reduced epochs to meet time constraint
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# ---  EVALUATION ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("--- Model Training Complete ---")
print(f"Test Accuracy: {accuracy*100:.2f}% (Baseline accuracy for Churn is typically ~73%)")
print(f"Test Loss: {loss:.4f}")
model.summary(print_fn=lambda x: print("ANN Model Summary:\n" + x.replace('\n', '\n  ')))
