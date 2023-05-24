import numpy as np
from sklearn import preprocessing
# Надання позначок вхідних даних
Input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
# Створимо кодувальника і встановимо відповіднсть міток числам
encoder = preprocessing.LabelEncoder ()
encoder.fit(Input_labels)
# Виведемо відображення
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)
# Перетворимо мітки за допомогою кодувальника
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list (encoded_values) )
# Декодуємо випадковий набір чисел
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\nDecoded labels =", list (decoded_list))
