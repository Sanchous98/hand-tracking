from __future__ import annotations
from functools import update_wrapper


class Debugger:
    def __init__(self, f_debugger=None, f_debug_before=None, f_debug_after=None, called=False):
        self.f_debugger = f_debugger
        self.f_debug_before = f_debug_before
        self.f_debug_after = f_debug_after
        self.called = called
        update_wrapper(self, self.f_debugger if self.debugged is None else self.debugged)

    @property
    def debugged(self) -> callable:
        return self.f_debug_before if self.f_debug_before is not None else self.f_debug_after

    def __call__(self, *args, **kwargs):
        if self.called:
            return self.debugged(*args, **kwargs)

        self.called = True

        if self.f_debug_before is not None:
            self.f_debugger(*args, **kwargs)
            result = self.f_debug_before(*args, **kwargs)
        if self.f_debug_after is not None:
            result = self.f_debug_after(*args, **kwargs)
            self.f_debugger(*args, **kwargs)

        self.called = False
        return result

    def before(self, recursive: bool = True):
        def wrapper(f_debug):
            return type(self)(self.f_debugger, f_debug, None, not recursive and self.called)

        return wrapper

    def after(self, recursive: bool = True):
        def wrapper(f_debug):
            return type(self)(self.f_debugger, None, f_debug, not recursive and self.called)

        return wrapper
