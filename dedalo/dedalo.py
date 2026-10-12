# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:35:51 2025

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es

DEDALO: Configurator for the Giulia.

The name comes from Daedalus, the legendary builder in Greek mythology. 
Like its namesake, this component sets clear values and options that allow 
Giulia to create and run custom simulations.

"""
# Configure Dedalo into path
import os
import sys
examples_dir = os.path.dirname(__file__)
root_dir = os.path.join(examples_dir, '..')
sys.path.insert(1, root_dir)

from abc import ABC, abstractmethod
import pandas as pd

from giulia.presets.g_standard_preset import Giulia as StandardGiulia
from giulia.presets.g_multi_frequency_layer_preset import Giulia as MultiLayerGiulia

from giulia.bs.bs_deployments import Network
from giulia.ue.ue_deployments import UE_Deployment
from typing import Any

from giulia.logger import warning, debug

# The df_ep consistency checks and repairs live with the rest of the test tooling; Dedalo
# just exposes them as methods bound to its own Giulia instance.
from tests.network_df_ep_check_list import run_all_checklist, print_checklist_results
from tests.network_df_ep_update import (
    run_all_updates,
    run_selected_updates,
    repair_dependants_of,
    updaters_for_field,
    UNREPAIRABLE_DEPENDANTS,
)


GiuliaLike = StandardGiulia | MultiLayerGiulia

# Provide shared Giulia accessors.
class Dedalo(ABC):
    """Base configurator that keeps a reference to the shared Giulia instance."""

    # Hold the Giulia orchestrator reference.
    def __init__(self, giulia: GiuliaLike) -> None:
        """Store the Giulia simulation object to be configured.

        Parameters
        ----------
        giulia: Giulia
            Simulation orchestrator that Dedalo utilities will manipulate.
        """
        self.giulia_obj = giulia

    @abstractmethod
    def process_static(self, episode_index: int = 0, **kwargs) -> None:
        pass

    @abstractmethod
    def process_realTime(self, step_index: int = 0, **kwargs) -> None:
        pass


    def set_giulia_network_df_ep(self,
                                name_cell_list: str | list,
                                name_attribute: str,
                                value_list: Any | list | list[Any],
                                repair: bool = True) -> None:
        """
        Update values for a specific attribute in the Giulia network deployment DataFrame (`df_ep`).

        Some `df_ep` columns are *derived* from others, so a bare write leaves its dependants
        stale and Giulia does not notice — see ``docs/dedalo/df-ep-consistency.md`` for a
        measured case where that silently doubled a reported KPI. `repair` closes that hole
        by default.

        Parameters
        ----------
        name_cell_list : str | list
            The name(s) of the network cell(s) to update. Can be a single string or a list of strings.
        name_attribute : str
            The column/attribute in the DataFrame to be updated.
        value_list : Any | list | list[Any]
            The value(s) to assign to the specified attribute. Can be a single value or a list of values
            (corresponding one-to-one with the provided `name_cell_list`).
        repair : bool, default True
            Recompute the derived columns that this write invalidates, via
            ``tests.network_df_ep_update.repair_dependants_of``. Targeted: only the dependants
            of `name_attribute` are touched, so it is a no-op for the majority of columns,
            which have none. Cheap (pandas arithmetic over the cell rows) and idempotent,
            since every updater recomputes from the current sources. Pass False to get the
            historical raw-write behaviour.

        Note
        ----
        This does not re-``process()`` the Giulia objects that cached the table at
        construction. When the written column feeds one of them (bandwidth, SCS, carrier
        frequency, array geometry) the caller must re-process it — as
        ``Dedalo_UnRL.set_BW_scs_prb_numerology`` does. Omitting it fails loudly with a shape
        mismatch, not silently.

        Raises
        ------
        TypeError
            If `network_deployment_obj` is not an instance of `Network`.
        ValueError
            If `network_deployment_obj.df_ep` is not a pandas DataFrame.
        KeyError
            If `name_attribute` is not a column in the DataFrame.

        Returns
        -------
        None
            This method updates the DataFrame in place and does not return anything.
        """

        # Validate that Giulia network deployment object is properly initialized
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list. Giulia has to be initialized first.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame. Giulia has to be initialized first.")

        # Ensure inputs are lists
        if isinstance(name_cell_list, str): 
            name_cell_list = [name_cell_list]

        if not isinstance(value_list, list): 
            value_list = [value_list]

        # Broadcast single value if needed
        if len(value_list) == 1 and len(name_cell_list) > 1:
            value_list = value_list * len(name_cell_list)

        # Final sanity check
        if len(name_cell_list) != len(value_list):
            raise ValueError(
                f"Length mismatch: {len(name_cell_list)} cells provided but {len(value_list)} values supplied."
            )

        # Get Giulia DataFrame Engineering Parameters
        df_ep = self.giulia_obj.network_deployment_obj.df_ep

        # Ensure the attribute exists in the DataFrame
        if name_attribute not in df_ep.columns:
            raise KeyError(f"Attribute '{name_attribute}' not found in Giulia deployment DataFrame.")

        # Update the specified attribute for each provided cell
        for name_cell, value in zip(name_cell_list, value_list):
            self.giulia_obj.network_deployment_obj.df_ep.loc[
                self.giulia_obj.network_deployment_obj.df_ep['name'] == name_cell,
                name_attribute
            ] = value

        # Recompute whatever this write invalidated, before anything can read it stale.
        if repair:
            updaters = updaters_for_field(name_attribute)
            if updaters:
                debug(f"set_giulia_network_df_ep: repairing dependants of '{name_attribute}': "
                      f"{[f.__name__ for f in updaters]}")
                self.giulia_obj.network_deployment_obj.df_ep = repair_dependants_of(
                    self.giulia_obj.network_deployment_obj.df_ep, name_attribute)

            # Checked but not repairable: warn rather than let it drift unannounced.
            if name_attribute in UNREPAIRABLE_DEPENDANTS:
                warning(
                    f"set_giulia_network_df_ep: '{name_attribute}' determines "
                    f"'{UNREPAIRABLE_DEPENDANTS[name_attribute]}', for which no updater exists. "
                    f"Recompute it yourself, or run check_df_ep() to confirm it is still consistent."
                )

        return

    def check_df_ep(self, verbose: bool = False) -> dict[str, tuple[bool, str]]:
        """Run the derived-field checklist against the bound instance's network ``df_ep``.

        Returns ``{check_name: (passed, message)}``. Cheap enough to call at the top of
        ``process_static`` during development.
        """
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        results = run_all_checklist(self.giulia_obj.network_deployment_obj.df_ep, verbose=False)
        if verbose:
            print_checklist_results(results)
        return results

    def assert_df_ep_consistent(self) -> None:
        """Raise ``ValueError`` if any derived column of the network ``df_ep`` is stale."""
        failed = {name: msg for name, (ok, msg) in self.check_df_ep().items() if not ok}
        if failed:
            raise ValueError(
                "df_ep is inconsistent — a derived column does not match its sources: "
                f"{sorted(failed)}\n" + "\n".join(failed.values())
            )

    def repair_df_ep(self, fields: list | None = None) -> None:
        """Recompute derived columns of the network ``df_ep``.

        Thin wrapper over ``tests.network_df_ep_update``. ``fields=None`` runs every updater
        (``run_all_updates``); otherwise pass updater functions or their names, as accepted by
        ``run_selected_updates``. Repairing the table does not re-``process()`` the Giulia
        objects that cached it — the caller must still do that.
        """
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")

        df_ep = self.giulia_obj.network_deployment_obj.df_ep
        self.giulia_obj.network_deployment_obj.df_ep = (
            run_all_updates(df_ep) if fields is None else run_selected_updates(df_ep, fields)
        )

    def set_giulia_ue_df_ep(self,
                            name_ue_list: str | list, 
                            name_attribute: str,
                            value_list: Any | list | list[Any]) -> None:
        """
        Update values for a specific attribute in the Giulia UE (User Equipment) deployment DataFrame (`df_ep`).

        Note
        ----
        Unlike :meth:`set_giulia_network_df_ep` this has no ``repair`` argument: the updaters
        in ``tests.network_df_ep_update`` are written against the *network* schema, and several of
        their source columns (``dl_bandwidth_MHz``, …) do not exist in the UE table. The UE
        columns Dedalo actually writes — ``ue_target_rate_Mbps``, ``traffic_generation_model``
        — have no derived dependants, so there is nothing to repair.
        """

        # Validate that Giulia UE deployment object is properly initialized
        if not isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment):
            raise TypeError("The ue_deployment_obj should be a UE_Deployment object. Giulia has to be initialized first.")
        if not isinstance(self.giulia_obj.ue_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The ue_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame. Giulia has to be initialized first.")

        # Ensure inputs are lists for consistent iteration
        if isinstance(name_ue_list, str):
            name_ue_list = [name_ue_list]
        if not isinstance(value_list, list):
            value_list = [value_list]

        # Broadcast single value if needed (same behavior as set_giulia_network_df_ep)
        if len(value_list) == 1 and len(name_ue_list) > 1:
            value_list = value_list * len(name_ue_list)

        # Final sanity check
        if len(name_ue_list) != len(value_list):
            raise ValueError(
                f"Length mismatch: {len(name_ue_list)} UEs provided but {len(value_list)} values supplied."
            )

        df_ep = self.giulia_obj.ue_deployment_obj.df_ep

        # Ensure the attribute exists in the DataFrame
        if name_attribute not in df_ep.columns:
            raise KeyError(f"Attribute '{name_attribute}' not found in Giulia UE deployment DataFrame.")

        # Update the specified attribute for each provided UE
        for name_ue, value in zip(name_ue_list, value_list):
            self.giulia_obj.ue_deployment_obj.df_ep.loc[
                self.giulia_obj.ue_deployment_obj.df_ep['name'] == name_ue, 
                name_attribute
            ] = value

        return

    
    def get_cell_names(self, attribute: str|None, value: str|None=None) -> list[str]:
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        # If no attribute is provided, return all cell names.
        if attribute is None:
            return self.giulia_obj.network_deployment_obj.df_ep['name'].tolist()
        else:
            return self.giulia_obj.network_deployment_obj.df_ep.loc[self.giulia_obj.network_deployment_obj.df_ep[attribute] == value, "name"].tolist()
    
    def get_ue_names(self, attribute: str|None, value: str|None=None) -> list[str]:
        if not isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment):
            raise TypeError("The ue_deployment_obj should be a UE_Deployment object, not a list.")
        if not isinstance(self.giulia_obj.ue_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The ue_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        # If no attribute is provided, return all UE names.
        if attribute is None:
            return self.giulia_obj.ue_deployment_obj.df_ep['name'].tolist()
        else:
            return self.giulia_obj.ue_deployment_obj.df_ep.loc[self.giulia_obj.ue_deployment_obj.df_ep[attribute] == value, "name"].tolist()
    
    @property
    def giulia_number_of_cells(self) -> int:
        """Retrieve the current number of cells from the Giulia deployment object.

        Returns
        -------
        int
            Current number of cells in the Giulia simulation.
        """
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        return self.giulia_obj.network_deployment_obj.df_ep.shape[0]
    @property
    def giulia_number_of_ue(self) -> int:
        """Retrieve the current number of UEs from the Giulia deployment object.

        Returns
        -------
        int
            Current number of UEs in the Giulia simulation.
        """
        if not isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment):
            raise TypeError("The ue_deployment_obj should be a UE_Deployment object, not a list.")
        if not isinstance(self.giulia_obj.ue_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The ue_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        return self.giulia_obj.ue_deployment_obj.df_ep.shape[0]


if __name__ == "__main__":
    print('This the module dedalo/dedalo.py. It is not meant to be run directly.')
