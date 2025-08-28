from my_perceptron import gera_amostra
from numpy.random import seed
from sklearn.linear_model import Perceptron as sk_perceptron

seed(42)

amostra = gera_amostra(20,0.5)
pontos = []
gabaritos = []
for ponto,gabarito in amostra:
    pontos.append(ponto)
    gabaritos.append(gabarito)

clf = sk_perceptron(tol=1e-3,random_state=42)
clf.fit(pontos,gabaritos)

score = clf.score(pontos,gabaritos)
print(f"score: {score}")
