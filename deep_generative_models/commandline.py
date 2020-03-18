import os
import signal
import sys

from typing import List, Dict, Any, Tuple


def parse_int_list(s: str, separator: str = ",") -> List[int]:
    if s is None or s == "":
        return []
    return [int(i) for i in s.split(separator)]


def parse_dict(s: str, separator: str = ",", key_value_separator: str = ":", value_type: type = str) -> Dict[str, Any]:
    result = dict()
    if s is not None and s != "":
        for key_value in s.split(separator):
            key, value = key_value.split(key_value_separator)
            result[key] = value_type(value)
    return result


def create_parent_directories_if_needed(file_path: str) -> str:
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
    return file_path


class DelayedKeyboardInterrupt:
    """
    The typing of signals and handlers is too complicated.
    """

    SIGNAL_NUMBERS = [signal.SIGINT, signal.SIGTERM]

    signals_received: List[Tuple]
    old_handler_by_signal_number: Dict

    def __init__(self):
        self.signals_received = []
        self.old_handler_by_signal_number = {}

    def __enter__(self):
        # discard saved information
        self.signals_received = []
        self.old_handler_by_signal_number = {}

        # listen to signals and keep the previous handlers
        for signal_number in self.SIGNAL_NUMBERS:
            self.old_handler_by_signal_number[signal_number] = signal.signal(signal_number, self.handler)

    def handler(self, signal_number, frame):
        # accumulate the received signal
        self.signals_received.append((signal_number, frame))
        print('Delaying received signal', signal_number)

    def __exit__(self, exception_type, exception_value, traceback):
        # restore signal handlers
        for signal_number, old_handler in self.old_handler_by_signal_number.items():
            signal.signal(signal_number, old_handler)

        # resume received signals
        for signal_number, frame in self.signals_received:
            print('Resuming received signal', signal_number)

            # get the old handler
            old_handler = self.old_handler_by_signal_number[signal_number]

            # call if possible
            if callable(old_handler):
                old_handler(signal_number, frame)

            # exit if needed
            elif old_handler == signal.SIG_DFL:
                sys.exit(0)

        # discard saved information
        self.signals_received = []
        self.old_handler_by_signal_number = {}

        # do not suppress exceptions
        return False
