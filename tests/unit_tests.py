"""
Created on Fri Apr 11 17:24:43 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""

import os
import sys
import unittest


# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from giulia.logger import LogLevel, set_log_level, get_log_level, __should_print as should_print


class UnitTests(unittest.TestCase):
    def test_logger_set_log_level(self):
        set_log_level(LogLevel.ERROR)
        assert get_log_level() == LogLevel.ERROR

    def test_logger___should_print(self):
        set_log_level(LogLevel.WARNING)

        assert should_print(LogLevel.ERROR, False)
        assert should_print(LogLevel.WARNING, False)
        assert not should_print(LogLevel.INFO, False)

        assert not should_print(LogLevel.ERROR, True)
        assert should_print(LogLevel.WARNING, True)
        assert not should_print(LogLevel.INFO, True)
