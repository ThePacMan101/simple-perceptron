from numpy.random import seed
import my_perceptron as P   

# seed(42)

amostra = P.gera_amostra(20,0.5)

perceptron = P.Perceptron()

perceptron.fit(amostra)

confusion_matrix = P.compute_confusion_matrix(amostra,perceptron)
VP,FP,FN,VN = confusion_matrix

f1 = P.f1_score(confusion_matrix)

print(f"f1 score: {f1}")
print(f"VN: {VN}")
print(f"FN: {FN}")
print(f"VP: {VP}")
print(f"FP: {FP}")