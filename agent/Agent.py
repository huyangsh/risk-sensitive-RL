from abc import abstractmethod

class Agent:
    @abstractmethod
    def train(self, T, verbose):
        raise NotImplementedError