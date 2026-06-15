from abc import ABC, abstractmethod

class BaseFilter(ABC):

    @abstractmethod
    def update(self, value, timestamp=None):
        pass
