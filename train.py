import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# --- Step 1: Generate realistic CarDekho-style dataset ---
np.random.seed(42)
n_samples = 500

car_data = {
    'Maruti': ['800', 'Alto', 'Swift', 'Baleno', 'Ciaz', 'Wagon R', 'Dzire', 'Ertiga', 'Vitara Brezza', 'Celerio'],
    'Hyundai': ['i10', 'i20', 'Verna', 'Creta', 'Venue', 'Santro', 'Xcent', 'Elantra', 'Tucson', 'Grand i10'],
    'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'Civic', 'CR-V', 'Brio', 'Accord', 'BR-V', 'Mobilio'],
    'Toyota': ['Innova', 'Fortuner', 'Corolla', 'Etios', 'Camry', 'Yaris', 'Glanza', 'Urban Cruiser', 'Vellfire', 'Etios Cross'],
    'Tata': ['Nexon', 'Harrier', 'Tiago', 'Tigor', 'Safari', 'Hexa', 'Nano', 'Bolt', 'Zest', 'Altroz'],
    'Ford': ['EcoSport', 'Figo', 'Endeavour', 'Aspire', 'Freestyle', 'Mustang', 'Ikon', 'Fiesta', 'Fusion', 'Mondeo'],
    'Mahindra': ['Scorpio', 'XUV500', 'Bolero', 'Thar', 'XUV300', 'Marazzo', 'KUV100', 'Verito', 'TUV300', 'Alturas G4'],
    'Volkswagen': ['Polo', 'Vento', 'Ameo', 'Tiguan', 'Passat', 'Jetta', 'T-Roc', 'Beetle', 'CrossPolo', 'Phaeton'],
}

# Base prices for companies (in lakhs)
company_base_price = {
    'Maruti': 3.5, 'Hyundai': 5.0, 'Honda': 6.0, 'Toyota': 8.0,
    'Tata': 4.5, 'Ford': 5.5, 'Mahindra': 6.5, 'Volkswagen': 7.0,
}

fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
fuel_weights = [0.45, 0.35, 0.10, 0.05, 0.05]
seller_types = ['Dealer', 'Individual']
transmission_types = ['Manual', 'Automatic']

rows = []
for _ in range(n_samples):
    company = np.random.choice(list(car_data.keys()))
    model = np.random.choice(car_data[company])
    car_name = f"{company} {model}"
    year = np.random.randint(2003, 2024)
    fuel_type = np.random.choice(fuel_types, p=fuel_weights)
    seller_type = np.random.choice(seller_types, p=[0.6, 0.4])
    transmission = np.random.choice(transmission_types, p=[0.7, 0.3])
    owner = np.random.choice([0, 1, 2, 3], p=[0.45, 0.35, 0.15, 0.05])
    kms_driven = np.random.randint(5000, 200000)

    # Calculate a realistic selling price
    base = company_base_price[company]
    age = 2024 - year
    depreciation = max(0.15, 1 - (age * 0.07))  # 7% depreciation per year
    price = base * depreciation

    # Adjustments
    if fuel_type == 'Diesel':
        price *= 1.15
    elif fuel_type == 'Electric':
        price *= 1.40
    elif fuel_type == 'CNG':
        price *= 0.90

    if transmission == 'Automatic':
        price *= 1.25

    if seller_type == 'Individual':
        price *= 0.92

    price *= max(0.5, 1 - (owner * 0.12))  # Owner depreciation
    price *= max(0.6, 1 - (kms_driven / 500000))  # Mileage depreciation

    # Add some noise
    price *= np.random.uniform(0.85, 1.15)
    price = round(price, 2)

    rows.append({
        'Car_Name': car_name,
        'Company': company,
        'Year': year,
        'Selling_Price': price,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Owner': owner,
        'Kms_Driven': kms_driven,
    })

df = pd.DataFrame(rows)
df.to_csv('cardekho_data.csv', index=False)
print(f"Generated dataset with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample data:\n{df.head()}")
print(f"\nPrice stats:\n{df['Selling_Price'].describe()}")

# --- Step 2: Preprocess and train ---
X = df.drop(columns=['Selling_Price', 'Car_Name'])
y = df['Selling_Price']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTrain R² Score: {train_score:.4f}")
print(f"Test  R² Score: {test_score:.4f}")

# --- Step 3: Save model and feature columns ---
BASE_DIR = Path(__file__).resolve().parent
joblib.dump(model, BASE_DIR / 'random_forest_model.pkl')
joblib.dump(X.columns.tolist(), BASE_DIR / 'feature_columns.pkl')

print(f"\nModel saved to: {BASE_DIR / 'random_forest_model.pkl'}")
print(f"Feature columns saved to: {BASE_DIR / 'feature_columns.pkl'}")
print("Done! You can now start the API server.")
