import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize


def calculate_empirical_rademacher_complexity(K: int,
                                              hypothesis_class,
                                              max_iter_maximization: float = 1000):
    training_data = hypothesis_class.training_data
    empirical_suprema_rademacher = []
    for _ in tqdm(range(K)):
        rademacher_rv = np.random.choice([-1, 1], len(training_data))
        res = minimize(fun=lambda parameter: -np.mean(rademacher_rv *
                                                      np.array([hypothesis_class.loss_function(training_point,
                                                                                               [training_point[0],
                                                                                                hypothesis_class(
                                                                                                    parameter=parameter,
                                                                                                    x=training_point[
                                                                                                        0])])
                                                                for training_point in training_data])),
                       x0=hypothesis_class.trained_parameter,  # set initial parameter the trained parameter
                       tol=1e-5,
                       constraints=hypothesis_class.constraints,
                       options={"disp": False, "maxiter": max_iter_maximization})
        if res.success:
            empirical_suprema_rademacher.append(-res.fun)
    #print(f"The approximated empirical rademacher complexity is: {np.mean(empirical_suprema_rademacher)}")
    return np.mean(empirical_suprema_rademacher)