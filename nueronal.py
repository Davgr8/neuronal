import sys
import numpy as np

class neuronal:
    def __init__(self):

        self.bia_salida = np.ones((2, 1))
        self.bia_oculto = np.ones((3, 1))
        self.bia_pesooculto = np.ones((3, 2))
        self.bia_pesosalida = np.ones((2, 3))

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))

    def sumamatrices(self,entradas):
        suma_pesooculto=np.dot(self.bia_pesooculto, entradas) + self.bia_oculto
        activacion=self.sigmoid(suma_pesooculto)

        suma_pesosalida=np.dot(self.bia_pesosalida, entradas) + self.bia_salida
        salida=self.sigmoid(suma_pesosalida)

        return salida

if __name__ == '__main__':
    funcio = neuronal()

    resp = np.array([0, 1]).reshape((2, 1))

    result = funcio.sumamatrices(resp)

    print("salida:",result)
