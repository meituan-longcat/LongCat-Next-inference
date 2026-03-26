import threading
import time


class LockTimeoutError(Exception):
    pass


class BaseTimeoutLock:
    def __init__(self, name, lock_class, default_timeout=10):
        self._name = name
        self._lock = lock_class()
        self._default_timeout = default_timeout
        self._is_locked = False

    def acquire(self, blocking=True, timeout=-1):
        """Acquire the lock with optional timeout."""
        t0 = time.time()
        if timeout == -1:
            timeout = self._default_timeout
        if not self._lock.acquire(blocking, timeout):
            raise LockTimeoutError(f"Could not acquire lock within {timeout} seconds")
        self._is_locked = True

    def release(self):
        """Release the lock."""
        if self._is_locked:
            self._lock.release()
            self._is_locked = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def acquire_with_timeout(self, timeout):
        """Acquire the lock with a custom timeout."""
        self.acquire(timeout=timeout)
        return self


class TimeoutLock(BaseTimeoutLock):
    def __init__(self, name, default_timeout=5):
        super().__init__(name, threading.Lock, default_timeout)


class TimeoutRLock(BaseTimeoutLock):
    def __init__(self, name, default_timeout=5):
        super().__init__(name, threading.RLock, default_timeout)
