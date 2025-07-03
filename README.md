# 🏠 Real Estate Price Prediction – Multiple Linear Regression

## 🚀 Purpose  
This project aims to predict real estate prices using **Multiple Linear Regression** in Python. It demonstrates the full lifecycle of a machine learning workflow including data preprocessing, visualization, model training, and evaluation — all applied to a real dataset of apartment listings.

## 🧠 Key Concepts Covered  
- Multiple linear regression  
- Exploratory data analysis  
- Data visualization (heatmaps, scatter plots)  
- Model training & testing with `scikit-learn`  
- Performance evaluation using `R² score`  
- Real-world prediction for new input data

## 🛠️ Technologies Used  
- Python 3.x  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn

## 📥 Dataset  
The dataset is loaded from this public URL:  
[📎 Dropbox XLSX file (Turkish housing data)](https://www.dropbox.com/s/luoopt5biecb04g/SATILIK_EV1.xlsx?dl=1)

**Features used for prediction:**
- `Oda_Sayısı` (Number of Rooms)  
- `Net_m2` (Net Square Meters)  
- `Katı` (Floor Number)  
- `Yaşı` (Age of the Building)

**Target variable:**
- `Fiyat` (Price in Turkish Lira)

## 📈 Sample Output  
- Correlation heatmap between all variables  
- Regression coefficients & model intercept  
- Side-by-side comparison of predicted vs. actual prices  
- R² scores for both training and test sets  
- Price prediction for a sample new house:
  - 3 rooms, 105 m², 4th floor, 8 years old

## 📦 How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
python real_estate_model.py

Note: You need internet access to load the dataset from Dropbox.

📊 What I Learned
	•	How to prepare real-world housing data for regression analysis
	•	How to interpret regression coefficients and model accuracy
	•	How to visualize correlations and prediction performance
	•	How to use the model to predict new, unseen housing data

📌 Next Steps
	•	Improve predictions by including more features (location, amenities, etc.)
	•	Replace missing values and normalize the dataset
	•	Deploy the model using Streamlit or Flask

🤝 Educational Context

This project was developed as a personal exercise to understand linear regression using real estate data. It is suitable for anyone learning applied machine learning with tabular data.

