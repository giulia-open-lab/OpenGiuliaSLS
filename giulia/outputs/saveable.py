"""
@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

from abc import ABCMeta, abstractmethod

from typing import List, Any, Optional

saveable_enabled = True

class Saveable(metaclass=ABCMeta):
    name: str
    """
    The name of the saveable object, typically the class name.
    """

    def __init__(self, register: bool = True):
        """
        Register this instance as a saveable object.

        Args:
            register: Whether this instance should be registered.
        """
        global saveable_enabled

        self.name = type(self).__name__

        # Check whether the saveable is already present by searching for its name
        if saveable_enabled and register and not Saveable.exists(self.name):
            # Only append if it doesn't already exist
            saveables.append(self)


    @staticmethod
    def enable():
        """
        Enables the saveable instance collection.
        """
        global saveable_enabled
        saveable_enabled = True


    @staticmethod
    def disable():
        """
        Disabled the saveable instance collection.

        Please note that this won't override any already registered saveables, it will just disable collecting new
        instances.

        Saveables work asynchronously, all the data is fetched when requested, not when the class is instantiated.
        """
        global saveable_enabled
        saveable_enabled = False


    @staticmethod
    def is_enabled() -> bool:
        """
        Check if the saveable instance collection is enabled.
        Returns:
            True if the saveable instance collection is enabled.
        """
        global saveable_enabled
        return saveable_enabled


    @staticmethod
    def exists(name: str) -> bool:
        """
        Check if a saveable object exists.
        Args:
            name: The name of the saveable object.
        Returns:
            True if the saveable object exists.
        """
        for obj in saveables:
            if obj.name == name:
                return True
        return False


    @abstractmethod
    def variables_list(self) -> List[str]:
        """
        Returns a list of variable names that should be saved from this object.

        This list must be constant. This method may be called multiple times during runtime, which may cause the output
        to be inconsistent.

        Default implementation returns an empty list.
        """
        return []


    def get_output(self, variable_name: str) -> Optional[Any]:
        """
        Fetches the actual value of the variable with the given name.
        Args:
            variable_name: The name of the variable.
        Returns:
            The value of the variable with the given name. Or ``None`` if it doesn't exist.
        """
        try:
            return getattr(self, variable_name)
        except AttributeError:
            return None


saveables: List[Saveable] = []
"""
This list contains all the instances of Saveable that have been created.
"""
