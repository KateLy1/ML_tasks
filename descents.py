import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        # Реализована формула затухающего шага обучения
        nu_k = self.lambda_ * ((self.s0 /(self.s0 + iteration)) ** self.p)
        return nu_k


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        #Изменила порядок, так как иначе в массиве loss_history первые две итерации будут одинаковыми "до обучения"
        self.iteration += 1
        weight_update = self.update_weights()
        #Хочу возвращать изменение весов из оптимизатора
        return weight_update


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        # Реализован vanilla градиентный спуск
        X_train = self.model.X_train
        y_train = self.model.y_train
        #Расчет градиента
        gradient = self.model.compute_gradients(X_train, y_train)
        #Обновление весов по формуле
        self.model.w = self.model.w - self.lr_schedule.get_lr(self.iteration) * gradient
        return - self.lr_schedule.get_lr(self.iteration) * gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        #Реализован стохастический градиентный спуск
        # 1) выбрать случайный батч (можно было реализовать с повторением индексов)
        random_indexes = [np.random.randint(0, self.model.X_train.shape[0]) for _ in range(self.batch_size)]
        X_train_B = self.model.X_train[random_indexes]
        #Смотрим как выбрать индексы у у в зависимости от типа
        if hasattr(self.model.y_train, 'iloc'): 
            y_train_B = self.model.y_train.iloc[random_indexes]
        else:
            y_train_B = self.model.y_train[random_indexes]
        # 2) вычислить градиенты на батче
        gradient = self.model.compute_gradients(X_train_B, y_train_B)
        # 3) обновить веса модели
        self.model.w = self.model.w - self.lr_schedule.get_lr(self.iteration) * gradient
        return -self.lr_schedule.get_lr(self.iteration) * gradient


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.avg_grad = None

    def update_weights(self):
        #Реализован SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            # Инициализированы хранилища при первом вызове нулями
            self.grad_memory = np.zeros((num_objects, num_features))
            self.avg_grad = np.zeros(num_features)

        # Выбор случайного индекса
        j = np.random.randint(0, num_objects)
        # Считаем градиент на среза
        g_j_new = self.model.compute_gradients(X_train[j:j+1], y_train[j:j+1])
        #Меняем средний градиент
        self.avg_grad = self.avg_grad + (g_j_new - self.grad_memory[j]) / num_objects
        #Обновляем память градиента
        self.grad_memory[j] = g_j_new
        #Обновляем веса
        self.model.w = self.model.w - self.lr_schedule.get_lr(self.iteration) * self.avg_grad
        return - self.lr_schedule.get_lr(self.iteration) * self.avg_grad


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        # Реализован градиентный спуск с моментумом
        X_train = self.model.X_train
        y_train = self.model.y_train
        #Инициализация вектора h
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        #Расчет градиента
        gradient = self.model.compute_gradients(X_train, y_train)
        #Считаем вектор h
        self.velocity = self.beta * self.velocity + self.lr_schedule.get_lr(self.iteration) * gradient
        #Обновляем веса
        self.model.w = self.model.w - self.velocity
        return - self.velocity


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        # Реализован Adam по формуле из ноутбука (все из теории)
        X_train = self.model.X_train
        y_train = self.model.y_train
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients(X_train, y_train)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_t = self.m / (1 - self.beta1 ** (self.iteration))
        v_t = self.v / (1 - self.beta2 ** (self.iteration))
        self.model.w = self.model.w - self.lr_schedule.get_lr(self.iteration) * m_t / (np.sqrt(v_t) + self.eps)
        return - self.lr_schedule.get_lr(self.iteration) * m_t / (np.sqrt(v_t) + self.eps)