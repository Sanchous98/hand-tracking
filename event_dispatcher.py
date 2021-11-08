from __future__ import annotations
from time import time
from typing import Dict, Union, Optional, Generator


class EventDispatcher:
    listeners = []

    @classmethod
    def add_listener(cls, listener: Listener):
        cls.listeners.append(listener)

    @classmethod
    def remove_listener(cls, listener: Listener):
        cls.listeners.remove(listener)

    @classmethod
    def has_listener(cls, event: Union[Event, str]) -> bool:
        for listener in cls.listeners:
            if event in listener.dispatches.keys() or isinstance(event,
                                                                 Event) and event.name in listener.dispatches.keys():
                return True

        return False

    @classmethod
    def get_listeners(cls, event: Optional[Union[Event, str]] = None) -> Generator[Listener]:
        if event is None:
            yield from cls.listeners

        if not cls.has_listener(event):
            return

        for listener in cls.listeners:
            if event in listener.dispatches.keys() \
                    or isinstance(event, Event) \
                    and event.name in listener.dispatches.keys():
                yield listener

    @classmethod
    def dispatch(cls, event: Event):
        if not cls.has_listener(event):
            return

        for listener in cls.listeners:
            if event in listener.dispatches.keys():
                listener.dispatches[event](event)
            elif event.name in listener.dispatches.keys():
                listener.dispatches[event.name](event)


class Event:
    def __init__(self):
        self.__stopped_propagation = False
        self.event_time = time()

    @property
    def stopped_propagation(self):
        return self.__stopped_propagation

    @stopped_propagation.setter
    def stopped_propagation(self, stop: bool):
        self.__stopped_propagation = stop

    @property
    def name(self) -> str:
        raise NotImplementedError


class Listener:
    @property
    def dispatches(self) -> Dict[Union[Event, str], callable]:
        raise NotImplementedError
