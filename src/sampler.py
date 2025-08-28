import numpy as np

def generate_samples(sample_size,standard_deviation,random_state = None):
    """
    Generates a list of points and their classifications
    according to 2 normal distributions, one centered at (0,0) 
    and the other centered at (1,1).

    sample_size:        the ammount of points to the generated
    
    standad_deviation:  the standard deviation of both normal
                        distributions
    
    random_state:       If not None, sets the numpy random seed to that
    
    returns:            sample,classes
    """
    if random_state is not None:
            np.random.seed(random_state)
    samples = []
    classes = []

    sample_size_00 = sample_size//2
    sample_size_11 = sample_size - sample_size_00
    for _ in range(sample_size_00):
        sample = np.random.normal(loc=np.array([0,0]),size=2,scale=standard_deviation)
        samples.append(sample)
        classes.append(-1)
    
    for _ in range(sample_size_11):
        sample = np.random.normal(loc=np.array([1,1]),size=2,scale=standard_deviation)
        samples.append(sample)
        classes.append(1)
    
    return samples,classes