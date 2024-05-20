# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:18:54 2024

@author: bahadir sahin
"""

# necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk

# Downloading the dataset and displaying the first five rows:
veri = pd.read_excel("https://www.dropbox.com/s/luoopt5biecb04g/SATILIK_EV1.xlsx?dl=1")
veri.head()

# Defining the target (y) and feature variables (X):
X = veri[['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]
y = veri['Fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the algorithm and creating the model based on the training data:
model = LinearRegression()
model.fit(X_train, y_train)

print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Öznitelik_Katsayıları'])
coeff_df

# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 400)
canvas1.pack()

label1 = tk.Label(root, text='Oda_Sayısı: ')
canvas1.create_window(100, 100, window=label1)
entry1 = tk.Entry(root)
canvas1.create_window(200, 100, window=entry1)

label2 = tk.Label(root, text='Net_M2: ')
canvas1.create_window(90, 120, window=label2)
entry2 = tk.Entry(root)
canvas1.create_window(200, 120, window=entry2)

label3 = tk.Label(root, text='Katı: ')
canvas1.create_window(80, 140, window=label3)
entry3 = tk.Entry(root)
canvas1.create_window(200, 140, window=entry3)

label4 = tk.Label(root, text='Yaşı: ')
canvas1.create_window(80, 160, window=label4)
entry4 = tk.Entry(root)
canvas1.create_window(200, 160, window=entry4)

def values():
    global Oda_Sayısı
    Oda_Sayısı = float(entry1.get())
    
    global Net_M2
    Net_M2 = float(entry2.get())
    
    global Katı
    Katı = float(entry3.get())
    
    global Yaşı
    Yaşı = float(entry4.get())
    
    input_data = pd.DataFrame([[Oda_Sayısı, Net_M2, Katı, Yaşı]], columns=['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı'])
    Prediction_result = ('Evin Tahmin Edilen Fiyati (₺): ', 1000*int(model.predict(input_data)))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='lawngreen')
    canvas1.create_window(200, 220, window=label_Prediction)

button1 = tk.Button (root, text='Evin Tahmin Fiyatini Hesapla', command=values, bg='orange')
canvas1.create_window(200, 190, window=button1)

root.mainloop()
