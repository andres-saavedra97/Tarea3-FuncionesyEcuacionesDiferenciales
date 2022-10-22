from sklearn.preprocessing import MinMaxScaler #Escala los mínimos y máximos en un rango de 0 a 1
from keras.models import Sequential #Un modelo secuencial para una pila simple de capas donde cada capa tiene un tensor de entrada y un tensor de salida.
from keras.layers import Dense #Una capa densa aplica pesos a todos los nodos de la capa anterior.
from numpy import asarray #Convierte la entrada en un arreglo.
from matplotlib import pyplot #Colección de funiones
import numpy as np #Para crear vectores y matrices
import re #Conjunto de strings 
np.set_printoptions(suppress=True) #Se establece la forma en qu elos arreglos, los apuntadores y otros objetos NumPy son mostrados.

x = asarray([i/500 for i in range(-500,500)])
y = asarray([np.power(4*i,3) + (2*i) + 1  for i in x]) #Se define la función.

x = x.reshape((len(x), 1)) #Cambia la forma del arreglo sin cambiar su contenido.
y = y.reshape((len(y), 1))

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)

model = Sequential() #Se implementa el modelo secuencial, utilizar tanh resulta en valores más grandes de gradiente durante el entrenamiento.
model.add(Dense(3, input_dim=1, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(2, input_dim=1, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') #Se define la función de pérdida y el optimizador.
model.fit(x, y, epochs=100, batch_size=10, verbose=1)
yhat = model.predict(x)

##Transformadas inversas
x_plot = scale_x.inverse_transform(x)
y_plot = scale_y.inverse_transform(y)
yhat_plot = scale_y.inverse_transform(yhat)

#Gráfico, se definen colores, leyendas y etiquetas.
pyplot.scatter(x_plot,y_plot, color='red', label='Solución Actual')
pyplot.scatter(x_plot,yhat_plot, color='black', label='Solución predecida')
pyplot.title(r'$y=x^3 + 2x +1')
pyplot.xlabel('Entrada x')
pyplot.ylabel('Salida y')
pyplot.legend()
pyplot.show()
