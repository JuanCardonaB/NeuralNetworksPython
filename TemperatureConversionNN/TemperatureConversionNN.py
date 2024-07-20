import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])


input_layer = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden_layer_1 = tf.keras.layers.Dense(units=3)
hidden_layer_2 = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([input_layer,hidden_layer_1, hidden_layer_2])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=100, verbose=False)
print("Modelo entrenado!")

# This make a graph of the loss function
# plt.xlabel("# Epoca")
# plt.ylabel("Magnitud de perdida")
# plt.plot(historial.history["loss"])
# plt.show()

print("Prediccion: ")
input_data = np.array([100.0], dtype=float)
resultado = modelo.predict(input_data)
print("El resultado es " + str(resultado) + " fahrenheit!")

# REGULAR PROGRAMATION
# def change (num):
#     return num * 1.8 + 32

# print(change(100))