from abc import ABC, abstractmethod
class ScheduleStrategy(ABC):
    @abstractmethod
    def get_config(self, epoch: int, metrics_history: dict = None):
        pass