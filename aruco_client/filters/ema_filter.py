from .base_filter import BaseFilter

class EMAFilter(BaseFilter):

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, value, timestamp=None):

        if self.value is None:
            self.value = value
            return value

        self.value = (
            self.alpha * value +
            (1.0 - self.alpha) * self.value
        )

        return self.value
