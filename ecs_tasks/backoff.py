"""Adaptive exponential backoff for AOSS throttling.

Thread-safe: can be shared across multiple writer threads.
Each thread calls wait() before sending, on_success/on_failure after response.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class AdaptiveBackoff:
    """Exponential backoff that increases on failure and decays on success.

    Compared to the old linear approach (+2/-1):
    - Increases faster on sustained throttling (exponential)
    - Recovers faster after throttling stops (exponential decay)
    """

    def __init__(self, initial=0.5, max_delay=30.0, increase_factor=2.0, decrease_factor=0.7):
        self.initial = initial
        self.max_delay = max_delay
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self._delay = 0.0
        self._lock = threading.Lock()

    @property
    def delay(self):
        with self._lock:
            return self._delay

    def on_failure(self):
        with self._lock:
            if self._delay == 0:
                self._delay = self.initial
            else:
                self._delay = min(self._delay * self.increase_factor, self.max_delay)
        logger.debug(f"Backoff increased to {self._delay:.2f}s")

    def on_success(self):
        with self._lock:
            self._delay = self._delay * self.decrease_factor
            if self._delay < 0.05:
                self._delay = 0.0

    def wait(self):
        delay = self.delay
        if delay > 0:
            time.sleep(delay)
