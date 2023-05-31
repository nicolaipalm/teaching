from abc import abstractmethod
from typing import Union

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint


class ParametricModel:
    @abstractmethod
    def __call__(self, parameter: np.ndarray, x: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def training_data(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, z_1, z_2) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def constraints(self) -> Union[NonlinearConstraint, LinearConstraint]:
        raise NotImplementedError

    def empirical_risk(self, parameter: np.ndarray):
        return np.mean(
            [self.loss_function(training_point, [training_point[0], self(parameter=parameter, x=training_point[0])]) for
             training_point in
             self.training_data])

    def train(self,
              initial_guess: np.ndarray,
              max_iter_training: int = 1000, ):  # ERM training
        print("Training...")
        res = minimize(fun=lambda parameter: self.empirical_risk(parameter),
                       x0=initial_guess,
                       tol=1e-5,
                       constraints=self.constraints,
                       options={"disp": True, "maxiter": max_iter_training})
        self.trained_parameter = res.x


class PenalizedLinearModel(ParametricModel):
    def __init__(self,
                 training_data: np.ndarray,
                 maximum_beta: float,
                 q_of_q_norm: int, ):
        self._maximum_beta = maximum_beta
        self._q_of_q_norm = q_of_q_norm
        self._training_data = training_data

    @property
    def training_data(self) -> np.ndarray:
        return self._training_data

    def __call__(self, parameter: np.ndarray, x: np.ndarray):
        return np.dot(parameter, x)

    def loss_function(self, z_1, z_2) -> float:
        y_1 = z_1[1]
        y_2 = z_2[1]
        return np.linalg.norm(y_1 - y_2) ** 2

    @property
    def constraints(self) -> NonlinearConstraint:
        return NonlinearConstraint(lambda x: np.sum(np.abs(x) ** self._q_of_q_norm) ** (1 / self._q_of_q_norm), -np.inf,
                                   self._maximum_beta)
