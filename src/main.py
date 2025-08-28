from perceptron import Perceptron as my_Perceptron
from sklearn.linear_model import Perceptron as sk_Perceptron
from sampler import generate_samples

iterations = 10
sample_size = 120
standard_deviation = 0.7

models = [
    ("Sci-Kit Learn's Perceptron" ,   sk_Perceptron),
    ("My Perceptron"              ,   my_Perceptron)
]

samples,classifications = generate_samples(sample_size,standard_deviation)

for model_name, model in models:
    mean_score = 0
    scores = []
    for _ in range(iterations):
        P = model()
        P.fit(samples,classifications)

        score = P.score(samples,classifications)
        scores.append(score)
        mean_score+=score
    
    mean_score /= iterations

    standard_deviation = 0
    for score in scores:
        standard_deviation+= (mean_score-score)**2
    standard_deviation = standard_deviation**(1/2)

    print( f"""
          Mean F1-Score from {model_name} = {mean_score}
          with Standard Deviation = {standard_deviation}
            """)
