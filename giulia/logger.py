import os
from enum import Enum
from typing import Optional, Iterable

# <editor-fold desc="ANSI Color Codes">
# Source: https://github.com/tartley/colorama/blob/master/colorama/ansi.py
CSI = '\033['

def code_to_chars(code):
    return CSI + str(code) + 'm'

class AnsiCodes(object):
    def __init__(self):
        # the subclasses declare class attributes which are numbers.
        # Upon instantiation we define instance attributes, which are the same
        # as the class attributes but wrapped with the ANSI escape sequence
        for name in dir(self):
            if not name.startswith('_'):
                value = getattr(self, name)
                setattr(self, name, code_to_chars(value))

class AnsiCursor(object):
    def UP(self, n=1):
        return CSI + str(n) + 'A'
    def DOWN(self, n=1):
        return CSI + str(n) + 'B'
    def FORWARD(self, n=1):
        return CSI + str(n) + 'C'
    def BACK(self, n=1):
        return CSI + str(n) + 'D'
    def POS(self, x=1, y=1):
        return CSI + str(y) + ';' + str(x) + 'H'

class AnsiFore(AnsiCodes):
    BLACK           = 30
    RED             = 31
    GREEN           = 32
    YELLOW          = 33
    BLUE            = 34
    MAGENTA         = 35
    CYAN            = 36
    WHITE           = 37
    RESET           = 39

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX   = 90
    LIGHTRED_EX     = 91
    LIGHTGREEN_EX   = 92
    LIGHTYELLOW_EX  = 93
    LIGHTBLUE_EX    = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX    = 96
    LIGHTWHITE_EX   = 97


class AnsiBack(AnsiCodes):
    BLACK           = 40
    RED             = 41
    GREEN           = 42
    YELLOW          = 43
    BLUE            = 44
    MAGENTA         = 45
    CYAN            = 46
    WHITE           = 47
    RESET           = 49

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX   = 100
    LIGHTRED_EX     = 101
    LIGHTGREEN_EX   = 102
    LIGHTYELLOW_EX  = 103
    LIGHTBLUE_EX    = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX    = 106
    LIGHTWHITE_EX   = 107


class AnsiStyle(AnsiCodes):
    BRIGHT    = 1
    DIM       = 2
    NORMAL    = 22
    RESET_ALL = 0

Fore   = AnsiFore()
Back   = AnsiBack()
Style  = AnsiStyle()
Cursor = AnsiCursor()
# </editor-fold>

class LogLevel(Enum):
    VERBOSE = 0
    """The maximum level of logging. Everything will be printed."""
    DEBUG = 1
    """A level intended for developers. Prints less info than VERBOSE, but still prints a lot."""
    INFO = 2
    """Prints some informational messages to keep track of what's happening. Loading bars will be shown up until this level."""
    WARNING = 3
    """Only prints warnings and soft-errors."""
    ERROR = 4
    """Only prints errors and crashes."""

    def __eq__(self, other):
        if not type(self) is type(other): return False
        return self.value == other.value

    def __lt__(self, other):
        if not type(self) is type(other): return False
        return self.value < other.value

    def __gt__(self, other):
        if not type(self) is type(other): return False
        return self.value > other.value

    def __le__(self, other):
        if not type(self) is type(other): return False
        return self.value <= other.value

    def __ge__(self, other):
        if not type(self) is type(other): return False
        return self.value >= other.value

default_log_level: LogLevel = LogLevel.WARNING
current_log_level: Optional[LogLevel] = None

def __should_print(level: LogLevel, exclusive: bool) -> bool:
    log_level: LogLevel = get_log_level()
    if exclusive: return log_level == level
    return level.value >= log_level.value

def __print(level: LogLevel, prefix_fore: int, prefix_back: int, fore: int, prefix: str, *values, exclusive: bool = False):
    if not __should_print(level, exclusive): return
    str_values: Iterable[str] = (str(value) for value in values)
    lines: str = ' '.join(str_values)
    lines = lines.replace('\n', '\n' + ' ' * (len(prefix) + 3))
    print(str(prefix_fore) + str(prefix_back) + f' {prefix} ' + str(Style.RESET_ALL) + str(fore) + ' ' + lines + str(Style.RESET_ALL))

def parse_log_level(level: str) -> Optional[LogLevel]:
    if level == 'VERBOSE' or level == 0:
        return LogLevel.VERBOSE
    elif level == 'DEBUG' or level == 1:
        return LogLevel.DEBUG
    elif level == 'INFO' or level == 2:
        return LogLevel.INFO
    elif level == 'WARNING' or level == 3:
        return LogLevel.WARNING
    elif level == 'ERROR' or level == 4:
        return LogLevel.ERROR
    else:
        return None

def set_log_level(level: Optional[LogLevel]):
    global current_log_level
    current_log_level = level

def get_log_level() -> LogLevel:
    global current_log_level
    if current_log_level is None:
        if 'LOG_LEVEL' in os.environ:
            level_value: str = os.environ['LOG_LEVEL']
            current_log_level = parse_log_level(level_value)
        else:
            current_log_level = default_log_level
    return current_log_level

def verbose(*values):
    """
    Prints a verbose message.
    :param values: the values to print
    """
    __print(LogLevel.VERBOSE, Fore.WHITE, Back.LIGHTBLACK_EX, Fore.LIGHTBLACK_EX, 'VERB', *values, exclusive=False)

def debug(*values, exclusive: bool = False):
    """
    Prints a debug message.
    :param values: the values to print
    :param exclusive: if ``True``, the text will only be printed if the debug level is exactly ``DEBUG``, and not on
    lower levels.
    """
    __print(LogLevel.DEBUG, Fore.WHITE, Back.LIGHTBLACK_EX, Fore.LIGHTBLACK_EX, 'DEBU', *values, exclusive=exclusive)

def info(*values, exclusive: bool = False):
    """
    Prints an info message.
    :param values: the values to print
    :param exclusive: if ``True``, the text will only be printed if the debug level is exactly ``INFO``, and not on
    lower levels.
    """
    __print(LogLevel.INFO, Fore.WHITE, Back.BLUE, Fore.BLUE, 'INFO', *values, exclusive=exclusive)
    
def info_logo(*values, exclusive: bool = False):
    """
    Prints an info message.
    :param values: the values to print
    :param exclusive: if ``True``, the text will only be printed if the debug level is exactly ``INFO``, and not on
    lower levels.
    """
    __print(LogLevel.INFO, Fore.WHITE, Back.MAGENTA, Fore.MAGENTA, 'INFO', *values, exclusive=exclusive)

def warning(*values, exclusive: bool = False):
    """
    Prints a warning message.
    :param values: the values to print
    :param exclusive: if ``True``, the text will only be printed if the debug level is exactly ``WARNING``, and not on
    lower levels.
    """
    __print(LogLevel.WARNING, Fore.WHITE, Back.YELLOW, Fore.YELLOW, 'WARN', *values, exclusive=exclusive)

def error(*values, exclusive: bool = False):
    """
    Prints an error message.
    :param values: the values to print
    :param exclusive: if ``True``, the text will only be printed if the debug level is exactly ``ERROR``, and not on
    lower levels.
    """
    __print(LogLevel.ERROR, Fore.WHITE, Back.RED, Fore.GREEN, 'ERRO', *values, exclusive=exclusive)
