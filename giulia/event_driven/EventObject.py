from abc import ABC, abstractmethod


class EventObject(ABC):
    @abstractmethod
    def process(self, rescheduling_us: int = -1) -> int:
        pass  # This is an abstract method, no implementation here.
