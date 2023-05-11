from abc import abstractmethod

class Env:
    @abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def eval(self, policy, T_eval):
        raise NotImplementedError