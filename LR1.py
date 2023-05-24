import numpy as np
from sklearn import preprocessing
import math
input_data = np.array([[5.1, -2.9, 3.3],
                      [-1.2, 7.8, -6.1],
                      [3.9, 0.4, 2.1],
                      [7.3, -9.9, -4.5]])

#Бінаризація даних
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\n Binarized data:\n", data_binarized)

#Виведення середнього значення та стандартного відхилення
print("\nBefore: ")
print("Mean=", input_data.mean(axis=0))
print("Std deviation = ", input_data.std(axis=0))

#Виключення середнього

data_scaled = preprocessing.scale(input_data)
print("\nAfter: ")
print("Mean=", data_scaled.mean(axis=0))
print("Std deviation = ", data_scaled.std(axis=0))

#Масштабування ознак

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range = (0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform (input_data)
print("\nMin max scaled date: \n", data_scaled_minmax)

#Нормалізація даних

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 normalized dataa: \n", data_normalized_l1)
print("\nl2 normalized dataa: \n", data_normalized_l2)

#Перевірка L1-нормалізації
print("L1 normanlization test:\n")
for row in data_normalized_l1:
    print(math.fsum(abs(row)))

#Перевірка L2-нормалізації
print("\nL2 normalization test:\n")
for row in data_normalized_l2:
    print(math.fsum(pow(row, 2)))