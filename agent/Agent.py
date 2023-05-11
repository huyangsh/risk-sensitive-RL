from abc import abstractmethod

class Agent:
    @abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, dataset, **kwargs):
        raise NotImplementedError