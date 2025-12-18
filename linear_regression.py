import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional, Union
import scipy

class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[Union[BaseDescent, str]] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []
        self.delta = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Реализована функцию предсказания в линейной регрессии (формула из теории)
        y_p = X @ self.w
        return y_p

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        if self.w is None:
            self.w = np.zeros(X.shape[1])
        
        if self.loss_function is LossFunction.MSE:
            # Реализовано вычисление градиентов для MSE (теоретическая формула)
            grad_Q = 2 * X.T @ (self.predict(X) - y) / y.shape[0]
            return grad_Q + self.l2_coef * self.w
            
        elif self.loss_function is LossFunction.MAE:
            # Реализовано вычисление градиентов для MAE (теоретическая формула)
            grad_Q = X.T @ np.sign(self.predict(X) - y) / y.shape[0]
            return grad_Q
            
        elif self.loss_function is LossFunction.LogCosh:
            # Реализовано вычисление градиентов для LogCosh (теоретическая формула)
            grad_Q = X.T @ np.tanh(self.predict(X) - y) / y.shape[0]
            return grad_Q
            
        elif self.loss_function is LossFunction.Huber:
            # Реализовано вычисление градиентов для Huber (теоретическая формула)
            errors = self.predict(X) - y
            if self.delta is None:
                delta = 1.35 * np.std(errors)   #Стандартное вычисление константы дельта (формула из открытых источников)
            comp = np.where(
                np.abs(errors) < delta,
                errors,
                delta * np.sign(errors)
            )
            grad_Q = comp / y.shape[0]
            return X.T @ grad_Q
            
        return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:

        if self.w is None:
            self.w = np.zeros(X.shape[1])
        
        if self.loss_function is LossFunction.MSE:
            # Реализована loss-функция MSE
            loss = np.mean((self.predict(X) - y)**2) + self.l2_coef * np.sum((self.w) ** 2) / 2
            self.loss_history.append(loss)
            return loss
            
        elif self.loss_function is LossFunction.MAE:
            # Реализована loss-функция MAE
            loss = np.mean(abs(self.predict(X) - y))
            self.loss_history.append(loss)
            return loss
            
        elif self.loss_function is LossFunction.LogCosh:
            # Реализована loss-функция LogCosh
            loss = np.mean(np.log(np.cosh((self.predict(X) - y))))
            self.loss_history.append(loss)
            return loss
            
        elif self.loss_function is LossFunction.Huber:
            # Реализована loss-функция Huber
            errors = self.predict(X) - y
            if self.delta is None:
                delta = 1.35 * np.std(errors)
            comp = np.where(
                np.abs(errors) < delta,
                (errors**2) / 2,
                delta * np.abs(errors) - (delta**2) / 2
            )
            loss = np.mean(comp)
            self.loss_history.append(loss)
            return loss
            
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.y_train = X, y
        #Инициализация весов
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        #Расчет и запись ошибки до обучения
        self.compute_loss(X, y)
        #Случай одного из 5 оптимизаторов
        if isinstance(self.optimizer, BaseDescent):
            self.optimizer.set_model(self)                  
            for _ in range(self.max_iter):
                #Обучаем модель, возвращаем изменение весов
                w_differance = self.optimizer.step()
                #Считаем ошибку после шага обучения
                self.compute_loss(X, y)
                #Критерии останова
                if np.sum((w_differance) ** 2) < self.tolerance or np.isnan(w_differance).any():
                    break
        #Если необходимо реализовать аналитическое решение
        elif self.optimizer is None:
            #Теоретическая формула
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
            #Считаем ошибку после обучения
            self.compute_loss(X, y)
        #SVD разложение
        elif self.optimizer == 'SVD':
            U, sing, V_T = scipy.sparse.linalg.svds(X, k=4)
            sing_op = np.diag(1 / sing)
            #Теоретическая формула
            self.w = V_T.T @ sing_op @ U.T @ y
            #Считаем ошибку после обучения
            self.compute_loss(X, y)
