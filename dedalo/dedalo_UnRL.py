colors_32 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd",
    "#31a354", "#756bb1", "#636363", "#e6550d", "#969696", "#6baed6", "#74c476", "#fd8d3c",
    "#9e9ac8", "#bdbdbd", "#fdd0a2", "#c7e9c0", "#fcbba1", "#dadaeb", "#c6dbef", "#fdae6b",
]

# Configure Dedalo into path
import os
import sys
examples_dir = os.path.dirname(__file__)
root_dir = os.path.join(examples_dir, '..')
sys.path.insert(1, root_dir)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from dedalo.dedalo import Dedalo
from giulia.presets.g_standard_preset import Giulia as StandardGiulia
from giulia.presets.g_multi_frequency_layer_preset import Giulia as MultiLayerGiulia
GiuliaLike = StandardGiulia | MultiLayerGiulia

from giulia.config.sim_config import Simulation_Config
from giulia.playground.sites import Site, Site_siteDeployment, Site_uePlaygroundDeployment

from giulia.bs.bs_deployments import Network
from giulia.ue.ue_deployments import UE_Deployment
from giulia.kpis.power_consumptions import Power_Consumption
from giulia.bs.base_stations import BaseStations

from giulia.logger import info, warning

from tests.network_df_ep_update import update_dl_PRBs_from_bandwidth_and_scs



class Dedalo_UnRL(Dedalo):
    """Energy-efficiency oriented Dedalo configurator for Giulia runs."""

    # Prepare the Dedalo EE instance and prime defaults.
    def __init__(self, giulia: GiuliaLike, **kwargs) -> None:
        """Prepare Dedalo for a Giulia simulation.

        Parameters
        ----------
        giulia: Giulia
            Simulation instance to be customised.
        **kwargs:
            Optional overrides passed to the UE initialisation helper.
        """
        super().__init__(giulia)

        # Set the starting hour for the episode.
        # default to hour 0 if not provided.
        self.start_hour = kwargs.get("start_hour", 0)  # default to hour 0 if not provided

        # Placeholder updated after querying Giulia runtime state.
        self.giulia_radio_status_dict: dict = {}

        # Initialize episode and step tracking for plotting
        self.episode_index: int | None = None
        self.step_index: int | None = None

        # Get the scenario model type from kwargs to determine the initialization path
        self.scenario_model = kwargs.get("scenario_model")
        
        if self.scenario_model == "data_driven_vlc":
            self._init_data_driven_vlc(**kwargs)
        else:
            self._init_legacy_giulia_scenario(**kwargs)

        # Set up color and marker dictionaries for plotting based on the new network deployment
        self._set_plots_dict()

        # Set Radio Unit parameters
        if kwargs.get("enable_ru_noisy_param", False): self.set_giulia_ru_random_parameters(**kwargs)

        return
    


    def _init_legacy_giulia_scenario(self, **kwargs) -> None:
        # Build the UE population and density profiles.
        self._init_number_of_ue(total_number_of_ue=6412) #  6412=(9*114)/0.16  #10000 # 12000

        # Set the Giulia radio unit ID different for each sector.
        self._set_giulia_radio_unit_id()

        # Compute the UE contour field for UE distribution
        self.ue_countour_field_xyz: tuple = self.make_ue_contour_field(seed=11)

        # self.plot_ue_contour_field(self.ue_countour_field_xyz, title="UE Contour Field")

        return


    def _init_data_driven_vlc(self, **kwargs) -> None:
        
        # Get Dataset to load for Data Driven scenario
        path_input_dp_ep: str|None = kwargs.get("data_set_path", None)

        assert path_input_dp_ep is not None, "Data set path must be provided."

        # Load DataSet
        input_df_ep: pd.DataFrame = pd.read_csv(path_input_dp_ep)

        # Make the text columns writable with non-text values.
        # From pandas 3, read_csv gives text columns the immutable Arrow-backed `str` dtype, which
        # rejects everything Dedalo and Giulia later write into them: integer radio unit IDs over
        # 'radio_unit_ID', paired cell lists and None over 'paired_cell', and so on. Widening those
        # columns to `object` once, here, keeps every later write behaving as it did on pandas 2.
        # (On pandas 2 the same columns already load as `object`, so this is a no-op there.)
        text_columns = [column for column in input_df_ep.columns if pd.api.types.is_string_dtype(input_df_ep[column])]
        input_df_ep = input_df_ep.astype({column: object for column in text_columns})

        # Basic check
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")

        # Update Giulia's network deployment with the loaded DataFrame
        self.giulia_obj.network_deployment_obj.df_ep = input_df_ep.copy()


        #### UPDATE NETWORK DEPLOYMENT IN GIULIA BASED ON THE LOADED DATASET ####
        self._update_giulia_NetworkObj_after_dataset_load()

        #### UPDATE SITE DEPLOYMENT IN GIULIA BASED ON THE LOADED DATASET ####
        self._update_giulia_siteDeploymentObj_after_dataset_load()

        #### UPDATE UE PLAYGROUND DEPLOYMENT IN GIULIA BASED ON THE LOADED DATASET ####
        self._update_giulia_uePlaygroundObj_after_dataset_load()

        # Build the UE population and density profiles.
        self._init_number_of_ue(total_number_of_ue=8606) #  8606=(9*153)/0.16  #10000 # 12000

        # Set unique radio unit IDs for each base station in Giulia
        self._set_giulia_radio_unit_id()

        # Reset Pairing
        # self._reset_giulia_pairing_cell_in_df_ep()

        # Update Giulia's internal objects based on the new network deployment
        self._update_giulia_object_static()

        return
    

    def get_giulia_N_UEsite_cell(self) -> int:
        """
        Retrieve the reference number of UE Site cells from the Giulia UE deployment object.
        """
        # Basic Checks
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj, Site):
            raise TypeError("The ue_playground_deployment_obj should be a Site object, not a list.")
        if isinstance(self.giulia_obj.ue_playground_deployment_obj, list):
            raise TypeError("The ref_number_of_cells should be an integer, not a list.")
        if self.giulia_obj.ue_playground_deployment_obj.ref_number_of_cells is None:
            raise ValueError("ref_number_of_cells is None. Ensure the UE deployment is properly configured.")

        return self.giulia_obj.ue_playground_deployment_obj.ref_number_of_cells
    
    # Build UE population information and cache initial Giulia densities.
    def _init_number_of_ue(self, **kwargs) -> None:
        """Initialise the UE population and per-hour density profile.

        Parameters
        ----------
        **kwargs:
            ``total_number_of_ue`` (int): desired daily UE population. Defaults to 2280.

        Raises
        ------
        TypeError
            If the Giulia deployment object is not a Network instance.
        ValueError
            If the deployment object does not expose a pandas DataFrame.
        """
        # Retrieve the total number of UEs to deploy. Defaults to 2280.
        self.total_number_of_ue = int(kwargs.get("total_number_of_ue", 2280))

        # Normalised UE proportion for each day hour.
        proportion_ue = np.asarray([0.13,  # 0h
                                    0.10,  # 1h
                                    0.07,  # 2h
                                    0.04,  # 3h
                                    0.03,  # 4h
                                    0.02,  # 5h
                                    0.02,  # 6h
                                    0.03,  # 7h
                                    0.04,  # 8h
                                    0.06,  # 9h
                                    0.08,  # 10h
                                    0.09,  # 11h
                                    0.10,  # 12h
                                    0.10,  # 13h
                                    0.11,  # 14h
                                    0.12,  # 15h
                                    0.12,  # 16h
                                    0.12,  # 17h
                                    0.13,  # 18h
                                    0.14,  # 19h
                                    0.15,  # 20h
                                    0.16,  # 21h
                                    0.16,  # 22h
                                    0.15,  # 23h
                                    0.13   # 24h
                                ])
        # Build a dictionary for easy hour-based access.
        self.proportion_ue_dict: dict = {f"hour_{h}": value for h, value in enumerate(proportion_ue)}

        # Get number of UE site cells from Giulia
        self.N_UEsite_cell = self.get_giulia_N_UEsite_cell()

        ue_per_UEsite_cell_func = lambda value, total_number_of_ue, N_UEsite_cell: int(round(value*(total_number_of_ue / N_UEsite_cell)))

        # Compute the uniform UE density per BS for each hour.
        self.ue_per_UEsite_cell_per_hour_dict = {hour: np.full(self.N_UEsite_cell, 
                                                                    ue_per_UEsite_cell_func(value, self.total_number_of_ue, self.N_UEsite_cell), 
                                                                    dtype=int) for hour, value in self.proportion_ue_dict.items()}

        return


    def _update_giulia_NetworkObj_after_dataset_load(self) -> None:

        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's network deployment object does not contain a valid DataFrame.")

        self.giulia_obj.network_deployment_obj.map_cellName_to_cellIndex = dict(zip(self.giulia_obj.network_deployment_obj.df_ep['name'], self.giulia_obj.network_deployment_obj.df_ep['ID']))
        self.giulia_obj.network_deployment_obj.map_cellIndex_to_cellName = dict(zip(self.giulia_obj.network_deployment_obj.df_ep['ID'], self.giulia_obj.network_deployment_obj.df_ep['name']))


        #### Set Coverage Capacity Cell role
        ####################################
        self.giulia_obj.network_deployment_obj.dl_freq_cov, self.giulia_obj.network_deployment_obj.dl_freq_cap =\
              self.giulia_obj.network_deployment_obj._compute_coverage_and_capacity_frequencies(self.giulia_obj.network_deployment_obj.df_ep)

        self.giulia_obj.network_deployment_obj.set_dfEp_coverage_capacity_cell_roles(self.giulia_obj.network_deployment_obj.dl_freq_cov, self.giulia_obj.network_deployment_obj.dl_freq_cap)

        # Init pairing field to None in df_ep
        self.giulia_obj.network_deployment_obj._init_to_None_cell_pairing_in_df_ep()

        #### Set carrier shutdown attributes
        self.giulia_obj.network_deployment_obj.set_dfEP_carrier_shutdown_attributes()


        #### Checks
        ########################
        self.giulia_obj.network_deployment_obj.basic_checks(self.giulia_obj.network_deployment_obj.df_ep)


    def _update_giulia_siteDeploymentObj_after_dataset_load(self) -> None:
        """Realign Giulia's site deployment reference layout with the freshly loaded network dataset."""

        if not isinstance(self.giulia_obj.site_deployment_obj, Site_siteDeployment):
            raise ValueError("Giulia's site deployment object is not a valid Site_siteDeployment instance.")  # site obj missing/wrong type
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj, Site_uePlaygroundDeployment):
            raise ValueError("Giulia's UE playground deployment object is not a valid Site_uePlaygroundDeployment instance.")  # UE obj needed for isd_m

        # NOTE: the scenario margin is taken from the UE playground's isd_m (not the site
        # deployment's own), preserving the original behaviour. Revisit if the site
        # deployment should use its own isd_m instead.
        isd_m = self.giulia_obj.ue_playground_deployment_obj.isd_m  # margin source: UE playground inter-site distance
        if isd_m is None:
            raise ValueError("UE playground isd_m is None; cannot compute scenario margin.")  # guard: isd_m must be set
        scenario_margin_isd_m = float(isd_m)  # coerce to float for the geometry maths

        # Apply the shared layout to the site deployment object
        self._apply_ref_layout_from_df_ep(
            self.giulia_obj.site_deployment_obj,  # target = site deployment
            scenario_margin_isd_m  # margin from UE playground isd_m
        )

        # Re-derive the site cell maps and cardinalities from df_ep: the Site object built them
        # from its own hexagonal layout, so they describe the scenario Giulia started from and
        # not the site cells that were just loaded.
        self._update_siteCells_maps_from_df_ep(self.giulia_obj.site_deployment_obj)


    def _update_giulia_uePlaygroundObj_after_dataset_load(self) -> None:
        """Realign Giulia's UE playground reference layout with the freshly loaded network dataset.

        Re-derives the shared cell-site geometry, then the UE-playground-specific
        cardinalities (sectors per site, number of UE-site cells and their maps), and
        propagates the new UE-site cell count into the simulation config (recomputes
        UEs per UE-site cell).
        """

        if not isinstance(self.giulia_obj.simulation_config_obj, Simulation_Config):
            raise ValueError("Giulia's simulation configuration object is not a valid Simulation_Config instance.")  # sim config needed downstream
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj, Site_uePlaygroundDeployment):
            raise ValueError("Giulia's UE playground deployment object is not a valid Site_uePlaygroundDeployment instance.")  # UE obj missing/wrong type

        ue_playground_obj = self.giulia_obj.ue_playground_deployment_obj  # local alias for readability
        isd_m = ue_playground_obj.isd_m  # UE playground inter-site distance (margin source)
        if isd_m is None:
            raise ValueError("UE playground isd_m is None; cannot compute scenario margin.")  # guard: isd_m must be set
        scenario_margin_isd_m = float(isd_m)  # coerce to float for the geometry maths

        # Shared cell-site geometry (positions, site count, scenario box, centre)
        self._apply_ref_layout_from_df_ep(ue_playground_obj, scenario_margin_isd_m)  # write common layout fields

        # UE-playground-specific cardinalities:
        # assume a default sectorization of 3 sectors per site, used to size the sector angular width
        ue_playground_obj.ref_number_of_sectors_per_site = 3  # default 3 sectors per site

        # Re-derive the UE-site cell maps and count from df_ep. The Site object built them from
        # its own hexagonal layout, so they describe the scenario Giulia started from and not the
        # site cells that were just loaded; sectors x sites would in turn count sectors that a
        # downsampled df_ep no longer contains.
        self._update_UEsiteCells_maps_from_df_ep(ue_playground_obj)

        # Propagate the new UE-site cell count into the simulation config (recomputes UEs per UE-site cell)
        self.giulia_obj.simulation_config_obj._set_ue_playground_deployment_obj(ue_playground_obj)  # hand the updated UE obj to sim config
        self.giulia_obj.simulation_config_obj._set_number_of_ues_per_UEsite_cell()  # recompute UEs per UE-site cell


    def _apply_ref_layout_from_df_ep(self, target_site_obj: Site_siteDeployment|Site_uePlaygroundDeployment, scenario_margin_isd_m: float) -> None:
        """Write the reference layout geometry shared by every Site deployment onto `target_site_obj`.

        Re-derives, from the loaded network df_ep, the cell-site positions, the site
        count, the scenario bounding box (site span + isd margin + 100 m on each axis)
        and the layout centre. Deployment-type-specific fields are left to the caller.

        Parameters
        ----------
        target_site_obj: Site
            The site deployment object to realign (site deployment or UE playground).
        scenario_margin_isd_m: float
            ISD-based margin added on each axis when sizing the scenario bounding box.
        """

        # Type checks: the underlying network df_ep must be valid before we touch the layout
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")  # network obj missing/wrong type
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's network deployment object does not contain a valid DataFrame.")  # df_ep not loaded yet

        # Extract the unique site coordinates from the loaded engineering-parameters DataFrame.
        # df_ep holds one row per *cell*, so the sectors (and any co-located carriers) of a site all
        # repeat that site's (x, y). A site is identified by the coordinate *pair*, so de-duplicate on
        # both columns jointly: calling .unique() column by column would break the x<->y pairing and
        # under-count sites as soon as two sites share an x or a y (any regular/grid layout).
        site_positions_m = self.giulia_obj.network_deployment_obj.df_ep[
            ["position_x_m", "position_y_m"]
        ].drop_duplicates().to_numpy(dtype=float)  # (N_sites, 2) [x, y], one row per distinct site

        site_positions_x = site_positions_m[:, 0]  # x of every distinct site
        site_positions_y = site_positions_m[:, 1]  # y of every distinct site

        # Store the (N_sites, 2) array as the reference cell-site layout
        target_site_obj.ref_cell_site_positions_m = site_positions_m  # (N_sites, 2) [x, y] matrix

        # Reference number of sites = number of unique positions
        target_site_obj.ref_number_of_cell_sites = target_site_obj.ref_cell_site_positions_m.shape[0]  # row count = site count

        # Scenario bounding box = site span + isd margin + 100 m on each axis, so UEs can be dropped around the edge sites
        target_site_obj.scenario_x_side_length_m = site_positions_x.max() - site_positions_x.min() + scenario_margin_isd_m + 100  # x extent + margin
        target_site_obj.scenario_y_side_length_m = site_positions_y.max() - site_positions_y.min() + scenario_margin_isd_m + 100  # y extent + margin

        # Layout centre = midpoint of the site span
        x_centre = (site_positions_x.max() - site_positions_x.min()) / 2 + site_positions_x.min()  # mid-x of the site span
        y_centre = (site_positions_y.max() - site_positions_y.min()) / 2 + site_positions_y.min()  # mid-y of the site span
        target_site_obj.layout_centre_m = (x_centre, y_centre, 0.00)  # store centre as (x, y, z=0)


    def _df_ep_and_site_index_per_siteCell(self, target_site_obj: Site_siteDeployment|Site_uePlaygroundDeployment) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the loaded df_ep and, for every site cell in it, the *row index* of its cell site.

        `map_cell_to_site` is consumed as an index into `ref_cell_site_positions_m` (see
        `giulia/ue/ue_deployments.py`, `calculate_ue_locations_at_random_per_site_using_hexagon`),
        so the raw `site_ID` cannot be used as such: Giulia's own scenarios happen to number
        sites 0..N-1, but a loaded dataset carries real-world identifiers (e.g. 4600068).
        `pd.factorize` therefore renumbers the sites 0..N-1 in order of first appearance, which
        is the same order in which `_apply_ref_layout_from_df_ep` de-duplicated the positions.

        Parameters
        ----------
        target_site_obj: Site_siteDeployment|Site_uePlaygroundDeployment
            The site deployment object whose `ref_cell_site_positions_m` the indices must address.
        """

        # Type checks: the underlying network df_ep must be valid before we touch the site cell maps
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")  # network obj missing/wrong type
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's network deployment object does not contain a valid DataFrame.")  # df_ep not loaded yet
        if not isinstance(target_site_obj.ref_cell_site_positions_m, np.ndarray):
            raise ValueError("ref_cell_site_positions_m is not set; run _apply_ref_layout_from_df_ep first.")  # layout must come first

        df_ep: pd.DataFrame = self.giulia_obj.network_deployment_obj.df_ep  # validated above

        # Renumber the sites 0..N-1 in order of first appearance in df_ep
        site_index_per_cell, unique_site_IDs = pd.factorize(df_ep["site_ID"])

        # A site is one row of ref_cell_site_positions_m, so the two counts must agree: they do
        # not when one site_ID sits at several positions (or two site_IDs share one position),
        # and every index into the positions array would then point at the wrong site.
        if len(unique_site_IDs) != target_site_obj.ref_cell_site_positions_m.shape[0]:
            raise ValueError(
                f"df_ep has {len(unique_site_IDs)} distinct site_ID but "
                f"{target_site_obj.ref_cell_site_positions_m.shape[0]} distinct site positions: "
                "site_ID and (position_x_m, position_y_m) must identify the same sites.")

        return df_ep, site_index_per_cell


    def _update_siteCells_maps_from_df_ep(self, target_site_obj: Site_siteDeployment|Site_uePlaygroundDeployment) -> None:
        """Rebuild the site cell maps and cardinalities of `target_site_obj` from the network df_ep.

        A *site cell* is one cell of the network deployment, i.e. one row of df_ep, taken in
        df_ep order: the row position is the site cell index, `map_cell_to_site` and
        `map_cell_to_bearing` are read straight off the site and bearing columns, and the
        cardinalities follow from the row count and the number of distinct cell sites.

        See :meth:`_update_UEsiteCells_maps_from_df_ep` for the UE-site cells of the UE
        playground, which are counted per sector rather than per row.

        Parameters
        ----------
        target_site_obj: Site_siteDeployment|Site_uePlaygroundDeployment
            The site deployment object to realign.
        """

        df_ep, site_index_per_siteCell = self._df_ep_and_site_index_per_siteCell(target_site_obj)  # validated df_ep + cell site row index of every site cell

        bearing_deg_per_siteCell = df_ep["antenna_config_hor_alpha_mec_bearing_deg"].to_numpy(dtype=int)  # mechanical bearing of every site cell

        target_site_obj.map_cell_to_site = {siteCell: int(site_index) for siteCell, site_index in enumerate(site_index_per_siteCell)}  # site cell index -> cell site row index
        target_site_obj.map_cell_to_bearing = {siteCell: float(bearing) for siteCell, bearing in enumerate(bearing_deg_per_siteCell)}  # site cell index -> bearing [deg]

        # Cardinalities implied by df_ep: one row per site cell, one cell site per distinct site_ID
        target_site_obj.ref_number_of_cells = len(df_ep)  # row count = site cell count
        if not isinstance(target_site_obj.ref_cell_site_positions_m, np.ndarray):
            raise ValueError("target_site_obj does not contain valid reference cell site positions.")
        target_site_obj.ref_number_of_cell_sites = target_site_obj.ref_cell_site_positions_m.shape[0]  # row count = site count


    def _update_UEsiteCells_maps_from_df_ep(self, ue_playground_obj: Site_uePlaygroundDeployment) -> None:
        """Rebuild the UE-site cell maps and cardinalities of `ue_playground_obj` from the network df_ep.

        A *UE-site cell* is a sector of the UE playground, not a row of df_ep: UEs are dropped
        once per sector, while a cell site carrying several frequency layers repeats each of its
        sectors once per carrier. The UE-site cells are therefore the distinct (cell site,
        bearing) pairs of df_ep, which keeps two properties the UE deployment relies on:

        - only the sectors that really exist get a UE population — a downsampled df_ep that cuts
          a cell site short no longer has UEs dropped into the sectors it lost;
        - a co-located carrier does not double the UE population of its cell site.

        Taken in order of first appearance the pairs are site-major and sector-minor, i.e. the
        UE-site cell ordering Giulia's own hexagonal playgrounds use.

        See :meth:`_update_siteCells_maps_from_df_ep` for the site cells of the network
        deployment, which are counted per df_ep row.

        Parameters
        ----------
        ue_playground_obj: Site_uePlaygroundDeployment
            The UE playground deployment object to realign.
        """

        df_ep, site_index_per_siteCell = self._df_ep_and_site_index_per_siteCell(ue_playground_obj)  # validated df_ep + cell site row index of every site cell

        # One UE-site cell per distinct (cell site, bearing): drop the co-located carrier repeats
        UEsiteCells_df = pd.DataFrame({
            "site_index": site_index_per_siteCell,
            "bearing_deg": df_ep["antenna_config_hor_alpha_mec_bearing_deg"].to_numpy(dtype=int),
        }).drop_duplicates()  # first appearance order: site-major, sector-minor

        ue_playground_obj.map_cell_to_site = {UEsiteCell: int(site_index) for UEsiteCell, site_index in enumerate(UEsiteCells_df["site_index"])}  # UE-site cell index -> cell site row index
        ue_playground_obj.map_cell_to_bearing = {UEsiteCell: float(bearing) for UEsiteCell, bearing in enumerate(UEsiteCells_df["bearing_deg"])}  # UE-site cell index -> bearing [deg]

        # Cardinalities implied by df_ep: one UE-site cell per sector, one cell site per distinct site_ID
        ue_playground_obj.ref_number_of_cells = len(UEsiteCells_df)  # sector count = UE-site cell count
        if not isinstance(ue_playground_obj.ref_cell_site_positions_m, np.ndarray):
            raise ValueError("Giulia's ue_playground_deployment_obj does not contain valid reference cell site positions.")
        ue_playground_obj.ref_number_of_cell_sites = ue_playground_obj.ref_cell_site_positions_m.shape[0]  # row count = site count

        return 


    def _set_giulia_radio_unit_id(self) -> None:
        """Assign unique radio unit IDs to each base station in Giulia."""
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        if not isinstance(self.giulia_obj.power_consumption_obj, Power_Consumption):
            raise TypeError("The power_consumption_obj should be a PowerConsumption object, not a list.")

        # Get Giulia DataFrame Engineering Parameters
        df_ep = self.giulia_obj.network_deployment_obj.df_ep

        # Assign unique radio unit IDs based on the ID.
        self.set_giulia_network_df_ep(name_cell_list=df_ep['name'].tolist(),
                                      name_attribute="radio_unit_ID",
                                      value_list=df_ep['ID'].tolist())

        ### RE-Run Power Consumption Process in Giulia for Updating.
        self.giulia_obj.power_consumption_obj.process()

        return


    def _reset_giulia_pairing_cell_in_df_ep(self) -> None:
        """Reset Giulia's pairing cell object in the DataFrame to reflect the new network deployment."""

        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        # Reset the 'paired_cell' column in the DataFrame to None for all entries
        self.giulia_obj.network_deployment_obj.df_ep['paired_cell'] = None

        # Reset carrier shutdown thresholds to None
        self.giulia_obj.network_deployment_obj.df_ep['carrier_shutdown_currentCell_usage_threshold'] = None
        self.giulia_obj.network_deployment_obj.df_ep['carrier_wakeUp_pairedCell__currentCell_usage_threshold'] = None

        return

    def _update_giulia_object_static(self, **kwargs) -> None:
        """
        Update Giulia's internal objects based on the new network deployment.
        This ensure that changes are reflected in the system before the simulation starts.

        Parameters
        ----------
        _g: Giulia
            Simulation instance to be updated.
        """
        # Skip updating pairing if specified in kwargs
        skip_pairing = kwargs.get("skip_pairing", False)

        object_list_name = [
        'ue_hotspot_deployment_obj',
        'pairing_cell_obj',
        'cell_antenna_array_structure_obj',
        'cell_SSB_conf_obj',
        'cell_CSI_RS_conf_obj',
        'power_consumption_obj',
        'cell_re_selection_conf_obj',
        'time_frequency_resource_obj',
        'mcs_resource_obj',
        'bler_vs_sinrs_lut_obj',
        'mi_obj' if self.giulia_obj.simulation_config_obj.sinr_mapping == "MIESM" else -1, # type: ignore
        'shadowing_map_cell_obj'
        ]
        for object_name in object_list_name:
            if object_name == -1:
                continue
            obj = getattr(self.giulia_obj, object_name)
            # if object_name == 'cell_SSB_conf_obj':
            #                 obj.TX_ver_active_number = np.full_like(obj.TX_ver_active_number, 2, dtype=int)
            #                 obj.TX_hor_active_number = np.full_like(obj.TX_hor_active_number, 4, dtype=int)
            # Apply configuration changes
            if object_name == 'pairing_cell_obj' and skip_pairing:
                continue
                
            info(f"Updating Giulia's {object_name} based on the new network deployment...")
            obj.process()
        return


    def _set_plots_dict(self) -> None:

        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's network deployment object does not contain a valid DataFrame.")

        self.color_dict = {
                        freq: colors_32[i]
                        for i, freq in enumerate(np.unique(self.giulia_obj.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy()))
                    }
        # RAT → marker shape for scatter plots (square = LTE, circle = NR)
        self.marker_rat_dict = {"LTE": "s", "NR": "o"}


    def plot_cells_for_frequencies(self, title_prefix="Network Layout"):
        """Plot the subset of `df` whose `dl_carrier_frequency_GHz` is in `target_freq_GHz_list`."""

        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's network deployment object does not contain a valid DataFrame.")
        # Get Giulia DataFrame Engineering Parameters
        df = self.giulia_obj.network_deployment_obj.df_ep

        # Identify unique frequencies in the DataFrame to plot
        target_freq_GHz_list = np.unique(df["dl_carrier_frequency_GHz"].to_numpy())

        # Extract arrays for plotting
        x_pos_m, y_pos_m, rat_arr, bst_deg_arr, freq_GHz_arr = self._extract_layout_arrays(df)

        fig = plt.figure(figsize=(6, 6))
        done_legend = {(freq, rat): False for freq in target_freq_GHz_list for rat in np.unique(rat_arr)}

        count = 0
        for target_freq_GHz in target_freq_GHz_list:
            mask = freq_GHz_arr == target_freq_GHz
            color = self.color_dict[target_freq_GHz]

            for x, y, rat, bst in zip(x_pos_m[mask], y_pos_m[mask], rat_arr[mask], bst_deg_arr[mask]):
                r = 150
                plt.arrow(x, y, r * np.cos(np.deg2rad(bst)), r * np.sin(np.deg2rad(bst)),
                        head_width=0.05, head_length=0.1, fc=color, ec=color)
                plt.scatter(x, y, marker=self.marker_rat_dict[rat], color=color, # type: ignore
                            s=100 if rat == "LTE" else 50)

                if not done_legend[(target_freq_GHz, rat)]:
                    plt.scatter([], [], color=color, marker=self.marker_rat_dict[rat], # type: ignore
                                label=f"{target_freq_GHz} GHz ({rat})")
                    done_legend[(target_freq_GHz, rat)] = True
                count += 1

        plt.legend(fontsize=7, framealpha=0.3)
        plt.grid(ls="--", alpha=0.5)
        plt.title(f"{title_prefix} - {', '.join(f'{f} GHz' for f in target_freq_GHz_list)}\n{count} cells")
        plt.show()


    def _extract_layout_arrays(self, df):
        """Return (x, y, RAT, bearing_deg, freq_GHz) arrays for plotting."""
        x = self._format_positions(df["position_x_m"].to_numpy())
        y = self._format_positions(df["position_y_m"].to_numpy())
        rat = df["RAT"].to_numpy()
        bst = df["antenna_config_hor_alpha_mec_bearing_deg"].to_numpy()
        freq = df["dl_carrier_frequency_GHz"].to_numpy()
        return x, y, rat, bst, freq

    def _format_positions(self, arr):
        """Convert position values (possibly stringy with multiple dots) into a float ndarray."""
        result = []
        for x in arr:
            if isinstance(x, str):
                result.append(
                    float(x.replace('.', '', x.count('.') - 1)) if x.count('.') > 1 else float(x)
                )
            else:
                result.append(float(x))
        return np.array(result)


    def set_BW_scs_prb_numerology(self, **kwargs) -> None:
        # Get bandwidth and subcarrier spacing values for 4G and 5G from kwargs, with defaults
        ## 4G
        BW_MHz_4G = kwargs.get("BW_MHz_4G", None)
        scs_kHz_4G = kwargs.get("scs_kHz_4G", None)
        cell_names_4G: list = self.get_cell_names("RAT", "LTE")
        ## 5G
        BW_MHz_5G = kwargs.get("BW_MHz_5G", None)
        scs_kHz_5G = kwargs.get("scs_kHz_5G", None)
        cell_names_5G: list = self.get_cell_names("RAT", "NR")

        # Each RAT is configured independently: a RAT is updated only if both its
        # bandwidth and subcarrier spacing are provided. This allows configuring
        # only 4G, only 5G, or both.
        configure_4G = BW_MHz_4G is not None and scs_kHz_4G is not None
        configure_5G = BW_MHz_5G is not None and scs_kHz_5G is not None

        if not configure_4G and not configure_5G:
            return  # Skip if neither RAT has a complete configuration

        # Update the Giulia network deployment DataFrame with the specified bandwidth and subcarrier spacing for 4G and 5G cells
        if configure_4G:
            self.set_giulia_network_df_ep(name_cell_list = cell_names_4G,
                                          name_attribute = "dl_bandwidth_MHz",
                                          value_list = [BW_MHz_4G] * len(cell_names_4G))
            self.set_giulia_network_df_ep(name_cell_list = cell_names_4G,
                                          name_attribute = "dl_subcarrier_spacing_kHz",
                                          value_list = [scs_kHz_4G] * len(cell_names_4G))

        if configure_5G:
            self.set_giulia_network_df_ep(name_cell_list = cell_names_5G,
                                    name_attribute = "dl_bandwidth_MHz",
                                    value_list = [BW_MHz_5G] * len(cell_names_5G))
            self.set_giulia_network_df_ep(name_cell_list = cell_names_5G,
                                    name_attribute = "dl_subcarrier_spacing_kHz",
                                    value_list = [scs_kHz_5G] * len(cell_names_5G))

        # Update the number of PRBs in Giulia's network deployment DataFrame based on the new bandwidth and subcarrier spacing values
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise ValueError("Giulia's network deployment object is not a valid Network instance.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's network deployment object does not contain a valid DataFrame.")
        self.giulia_obj.network_deployment_obj.df_ep = update_dl_PRBs_from_bandwidth_and_scs(self.giulia_obj.network_deployment_obj.df_ep)

        # Update Giulia's internal objects to reflect the new bandwidth, subcarrier spacing, and PRB configurations
        self._update_giulia_object_static(skip_pairing=True)
        return


    def _set_rescheduling_info(self) -> None:

        update_event_scheduling_info = {
            "snapshot_control": -1,
            "eventDriven_obj": -1,
            "ue_deployment_obj": -1,
            "ue_antenna_array_structure_obj": -1,
            "channel_sn_obj": -1,
            "distance_angles_ue_to_cell_obj": -1,
            "distance_angles_ueAnt_to_cellAnt_obj": -1,
            "antenna_pattern_gain_ue_to_cell_obj": -1,
            "array_steering_vector_ue_to_cell_obj": -1,
            "LOS_probability_ue_to_cell_obj": -1,
            "K_factor_ue_to_cell_obj": -1,
            "path_loss_ue_to_cell_obj": -1,
            "o2i_penetration_loss_ue_to_cell_obj": -1,
            "shadowing_gain_ue_to_cell_obj": -1,
            "slow_channel_gain_ue_to_cell_obj": -1,
            "los_channel_gain_ue_to_cell_obj": -1,
            
            # "time_frequency_resource_obj": 
            #     {"update_ue_carrier_info":-1},
                
            "dl_fast_fading_gain_ueAnt_to_cellAnt_obj": -1,
            "channel_gain_ue_to_cell_obj": -1,
            "SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj": -1,
            "SSB_precoded_channel_gain_ue_to_cell_obj": -1,
            "CSI_RS_precoded_channel_gain_ue_to_cell_obj": -1,
            "dl_noise_ue_to_cell_obj": -1,
            # "SSB_RSS_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "CSI_RS_RSS_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "CRS_RSRP_no_fast_fading_ue_to_cell_obj": self.dflt_rescheduling_us,
    
            # "SSB_RSRP_no_fast_fading_ue_to_cell_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "set_RSRP_ue_to_cell_dBm": self.dflt_rescheduling_us
            # },
    
            # "SSB_RSRP_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "CSI_RS_RSRP_ue_to_cell_obj": self.dflt_rescheduling_us,
    
            # "best_serving_cell_ID_per_ue_based_on_CRS_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "calculate_server_stats": self.dflt_rescheduling_us
            # },
    
            # "best_serving_cell_ID_per_ue_based_on_SSB_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "calculate_server_stats": self.dflt_rescheduling_us,
            # },
    
            # "cell_reselection_based_on_CRS_obj": self.dflt_rescheduling_us,
            # "cell_reselection_based_on_SSB_obj": self.dflt_rescheduling_us,
            # "best_serving_CSI_RS_per_ue_obj": self.dflt_rescheduling_us,
            # "CRS_sinr_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "SSB_sinr_ue_to_cell_obj": self.dflt_rescheduling_us,

            # "base_stations_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "mac": self.dflt_rescheduling_us,
            #     "carrier_shutdown": self.dflt_rescheduling_us,
            # },
            

            # "CSI_RS_sinr_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj": self.dflt_rescheduling_us,
            # "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj": self.dflt_rescheduling_us,
            # "performance_obj": self.dflt_rescheduling_us,
            # "ue_mobility_obj": self.dflt_rescheduling_us,
            # "bs_mobility_obj": self.dflt_rescheduling_us
        }

        assert isinstance(self.giulia_obj.simulation_config_obj, Simulation_Config), \
            "simulation_config_obj must be a Simulation_Config instanciated."
        
        # Update the event scheduling info in Giulia's simulation configuration.
        for key, value in update_event_scheduling_info.items():
            self.giulia_obj.simulation_config_obj.event_scheduling_info[key] = value

        return
    

    def set_pairing_df_ep_as_per_terrestrial_haps_hierarchical(self, df_ep_input: pd.DataFrame) -> None:
        """
        Update Giulia's df_ep pairing information based on df_ep_input.
    
        This function:
        - Adds the HAPS index to paired_cell lists (except for capacity cells).
        - Performs basic type checks.
        - Updates df_ep.paired_cell in the Giulia-associated Network.
        - Cleans up HAPS paired cells so that only coverage cells remain.
        - Sets the "cell_role" attribute to "coverage_capacity" for coverage cells.
        """

        # Basic Checks
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                "The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        # take the last index of self.giulia_obj.network_deployment_obj.df_ep,
        # as it correspond to the HAPS
        index_haps_in_df_ep: int = self.giulia_obj.network_deployment_obj.df_ep.shape[0] - 1
    
        # Get cell names to modify in dedalo associated Giulia. Get cell names from df_ep_input
        name_cell_list: list = df_ep_input["name"].tolist()
        # Get paired cell values to set in dedalo associated Giulia. Get paired_cell from df_ep_input
        value_list_paired_cell_list: list = df_ep_input["paired_cell"].tolist()
    
        # Get 4G Cells which would have been coverage cells
        coverage_mask = df_ep_input["cell_role"] == "coverage"
        coverage_cell_name_list: list = df_ep_input.loc[coverage_mask, "name"].tolist()
        coverage_cell_indices: list = df_ep_input.index[coverage_mask].tolist()
    
        # Get 5G Cells which would have been capacity cells
        capacity_mask = df_ep_input["cell_role"] == "capacity"
        capacity_cell_name_list: list = df_ep_input.loc[capacity_mask, "name"].tolist()
        capacity_cell_indices: list = df_ep_input.index[capacity_mask].tolist()
    
        # Add the HAPS ID to each paired cell list. Work on copies so df_ep_input is never
        # mutated: these copies (not the input's list objects) are what we store in Giulia
        # below, keeping the input DataFrame fully isolated from this update.
        paired_cell_updated: list = []
        for name, value in zip(name_cell_list, value_list_paired_cell_list):
            new_value = None if value is None else list(value)
            # Capacity cells should be paired only with 4G coverage cells, not the HAPS.
            if new_value is not None and name not in capacity_cell_name_list:
                new_value.append(index_haps_in_df_ep)
            paired_cell_updated.append(new_value)

        # Update paired cell values in dedalo associated Giulia from the copies above
        for name_cell, value in zip(name_cell_list, paired_cell_updated):
            # Get index of the cell to update in dedalo associated Giulia
            idx = self.giulia_obj.network_deployment_obj.df_ep.index[self.giulia_obj.network_deployment_obj.df_ep["name"] == name_cell][0]
            # Update
            self.giulia_obj.network_deployment_obj.df_ep.at[idx, "paired_cell"] = value

        # Eliminate from paired cell of HAPS
        haps_paired_cells: list = self.giulia_obj.network_deployment_obj.df_ep["paired_cell"][index_haps_in_df_ep]
        haps_paired_cells_aux = haps_paired_cells.copy()
    
        for cell in haps_paired_cells_aux:
            if cell in coverage_cell_indices:
                continue
            if cell in capacity_cell_indices:
                haps_paired_cells.remove(cell)
        del haps_paired_cells_aux
    
        # Check that haps_paired_cells contains the same entries as coverage_cell_indices
        for cell in coverage_cell_indices:
            if cell not in haps_paired_cells:
                raise KeyError(
                    "Error in setting paired cells in dedalo associated Giulia from df_ep_input. "
                    "HAPS paired cells do not contain all coverage cells."
                )
    
        # Set new label in dedalo associated Giulia df_ep for 4G cells
        self.set_giulia_network_df_ep(
            name_cell_list=coverage_cell_name_list,
            name_attribute="cell_role",
            value_list="coverage_capacity",
        )
    
        return
    

    def set_thresholds_df_ep_as_per_terrestrial_haps_hierarchical(self, df_ep_input: pd.DataFrame) -> None:
        """
        Sync carrier shutdown and wake-up thresholds in Giulia's df_ep
        based on the provided df_ep_input configuration.
        """

        # Basic Checks
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                "The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        # Get shutdown and wakeup thresholds
        shutdown_th = df_ep_input.loc[df_ep_input["cell_role"] == "capacity", "carrier_shutdown_currentCell_usage_threshold"].tolist()
        wakeup_th = df_ep_input.loc[df_ep_input["cell_role"] == "coverage", "carrier_wakeUp_pairedCell__currentCell_usage_threshold"].tolist()
        
        assert np.unique(wakeup_th).shape[0] == 1, (
            f"Expected exactly one wake-up threshold across coverage cells, got {np.unique(wakeup_th).tolist()} "
            f"(an empty list means df_ep_input has no 'coverage' cells)."
        )
        assert np.unique(shutdown_th).shape[0] == 1, (
            f"Expected exactly one shutdown threshold across capacity cells, got {np.unique(shutdown_th).tolist()} "
            f"(an empty list means df_ep_input has no 'capacity' cells)."
        )
        shutdown_th = shutdown_th[0]
        wakeup_th = wakeup_th[0]
        
        # Get all names to modify in final dedalo associated Giulia df_ep
        mask_coverage_df_giulia = (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "coverage") | (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "coverage_capacity")
        coverage_cell_name_list_df_giulia: list = self.giulia_obj.network_deployment_obj.df_ep.loc[mask_coverage_df_giulia, "name"].tolist()

        mask_capacity_df_giulia = (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "capacity") | (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "coverage_capacity")
        capacity_cell_name_list_df_giulia: list = self.giulia_obj.network_deployment_obj.df_ep.loc[mask_capacity_df_giulia, "name"].tolist()

        # Set the wake-up threshold on coverage cells and the shutdown threshold on capacity cells
        self.set_giulia_network_df_ep(
            name_cell_list=coverage_cell_name_list_df_giulia,
            name_attribute="carrier_wakeUp_pairedCell__currentCell_usage_threshold",
            value_list=wakeup_th,
        )
        self.set_giulia_network_df_ep(
            name_cell_list=capacity_cell_name_list_df_giulia,
            name_attribute="carrier_shutdown_currentCell_usage_threshold",
            value_list=shutdown_th,
        )

        return 



    def _get_ue_per_UEsite_cell_per_specific_hour(self, hour: int) -> np.ndarray:
        """Return the uniform UE density per UE site cell for a given hour of the day.

        Parameters
        ----------
        hour: int
            Hour index to sample. Values wrap around the 24-hour profile.

        Returns
        -------
        np.ndarray
            Array of UEs per UE site cell for the requested hour.
        """
        # Wrap around the 24-hour profile.
        hour = hour % 24

        return self.ue_per_UEsite_cell_per_hour_dict[f"hour_{hour}"]
    

    def set_TN_only_adjacency_neighbouring_matrix(self, matrix: np.ndarray) -> None:
        self._TN_only_adjacency_neighbouring_matrix: np.ndarray = matrix
        self._number_TN_only_cells = matrix.shape[0]
        assert matrix.shape[0] == matrix.shape[1], "Adjacency matrix must be square."
        return

    def set_in_Giulia_ue_per_UEsite_cell(self, value_per_UEsite_cell: np.ndarray) -> None:
        if isinstance(self.giulia_obj.simulation_config_obj, list):
                raise TypeError("The simulation_config_obj should be a Network object, not a list. giulia has to be intiialized")
        
        if isinstance(self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict, dict):  # Case for 38901 and colocated/nocolocated"
            # Split the UE count between hotspot and non-hotspot cells if the UE distribution includes a hotspot.
            # split_hotspot_and_nonhotspot: int = 2 if "hotspot" in self.giulia_obj.simulation_config_obj.ue_distribution else 1
            # self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] = value_per_UEsite_cell // split_hotspot_and_nonhotspot
            # self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_polygon"] = value_per_UEsite_cell // split_hotspot_and_nonhotspot

            if "hotspot" in self.giulia_obj.simulation_config_obj.ue_distribution:
                split_hotspot_and_nonhotspot: int = 2 if "hotspot" in self.giulia_obj.simulation_config_obj.ue_distribution else 1
                self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] = value_per_UEsite_cell // split_hotspot_and_nonhotspot
                self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_polygon"] = value_per_UEsite_cell // split_hotspot_and_nonhotspot
            else:
                self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] = value_per_UEsite_cell
                self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_polygon"] = None

        else:
            self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell = value_per_UEsite_cell



    def set_giulia_ru_random_parameters(self, **kwargs) -> None:

        """
        Set random parameters for Giulia's radio units.
        """

        if not isinstance(self.giulia_obj.power_consumption_obj, Power_Consumption):
            raise TypeError("The power_consumption_obj should be a PowerConsumption object, not a list.")
        
        # Get the number of radio units deployed from Giulia's power consumption object
        n_ru = self.giulia_obj.power_consumption_obj.number_of_radio_units_deployed

        # Get current radio unit parameters from Giulia's power consumption object
        radio_unit_parameters_d = {
                                    
                                    "P_0": np.asarray(
                                        self.giulia_obj.power_consumption_obj.P_0,
                                        dtype=object
                                    ).copy(),
                                    "number_of_TX_transceivers": np.asarray(
                                        self.giulia_obj.power_consumption_obj.number_of_TX_transceivers
                                    ).copy(),
                                    "number_of_RF_chains_per_TX_transceiver": np.asarray(
                                        self.giulia_obj.power_consumption_obj.number_of_RF_chains_per_TX_transceiver
                                    ).copy(),
                                    "consumption_per_RF_chain_W": np.asarray(
                                        self.giulia_obj.power_consumption_obj.consumption_per_RF_chain_W,
                                        dtype=float
                                    ).copy(),
                                    "consumption_per_MCPA_W": np.asarray(
                                        self.giulia_obj.power_consumption_obj.consumption_per_MCPA_W,
                                        dtype=float
                                    ).copy(),
                                    "efficiency_of_MCPA": np.asarray(
                                        self.giulia_obj.power_consumption_obj.efficiency_of_MCPA,
                                        dtype=float
                                    ).copy(),
                                }
        

        # Relative Noise Configuration
        rng = np.random.default_rng(seed=1010)

        relative_noise_mean = 0.0
        relative_noise_std = 0.5  # 50% relative Gaussian noise

        def relative_noise_factor(size,
                                  mean=relative_noise_mean,
                                  standard_deviation=relative_noise_std)-> np.ndarray:
            """
            Generate multiplicative Gaussian noise factors.

            For example, with standard_deviation=0.1,
            the factors are generally around 1 ± 10%.
            """
            relative_error = rng.normal(loc=mean, scale=standard_deviation, size=size)
            # Clip the factors to be within [0.5, 1.5] to avoid extreme values
            return np.clip(1.0 + relative_error,
                           0.5, 1.5)
        
        def apply_relative_noise(values, mean=relative_noise_mean, standard_deviation=relative_noise_std) -> np.ndarray:
            """Apply multiplicative relative noise to an array."""
            values = np.asarray(values, dtype=float)
            # Generate multiplicative noise factors based on the shape of the input values
            factors = relative_noise_factor(size=values.shape, mean=mean, standard_deviation=standard_deviation)
            return values * factors
        

        # Modify Static Power P0
        P_0_basic = radio_unit_parameters_d["P_0"]
        P_0_states = list(P_0_basic[0].keys())

        # The "active" state is noised as usual. Every other state must stay
        # <= the active value, so it is derived from the *noisy* active value
        # scaled by a logistic-normal factor in (0, 1):
        #     state = active_noisy * 1 / (1 + exp(-normal(0, 1)))
        ACTIVE_STATE = "active"

        if ACTIVE_STATE not in P_0_states:
            raise KeyError(
                f"Expected an '{ACTIVE_STATE}' state in P_0, "
                f"found states: {P_0_states}."
            )

        def logistic_normal_factor(size, mean=0.0, standard_deviation=1.0) -> np.ndarray:
            """
            Multiplicative factors in (0, 1) from a logistic-normal
            distribution, i.e. the sigmoid of a Gaussian. Used to shrink the
            non-active P_0 states below the active one while keeping them random.
            """
            return 1.0 / (1.0 + np.exp(-rng.normal(loc=mean, scale=standard_deviation, size=size)))

        # One relative-noise factor per radio unit for the active state
        P_0_active_factors = relative_noise_factor(n_ru)

        # One independent logistic-normal factor per radio unit for every
        # non-active state, so each stays <= the (noisy) active value
        non_active_states = [state for state in P_0_states if state != ACTIVE_STATE]
        P_0_shutdown_factors = {state: logistic_normal_factor(n_ru) for state in non_active_states}

        P_0_noisy = np.array(
            [
                {
                    ACTIVE_STATE: float(p_0[ACTIVE_STATE]) * P_0_active_factors[ru_index],
                    **{
                        state: float(p_0[ACTIVE_STATE])
                        * P_0_active_factors[ru_index]
                        * P_0_shutdown_factors[state][ru_index]
                        for state in non_active_states
                    },
                }
                for ru_index, p_0 in enumerate(P_0_basic)
            ],
            dtype=object
        )

        # Check
        for p0 in P_0_noisy: 
            assert p0['active'] - p0['carrier_shutdown'] >=0, \
                "P_0_noisy 'active' state must be greater than or equal to 'carrier_shutdown' state."

        # Noisy Parameters
        consumption_per_RF_chain_W_noisy = apply_relative_noise(self.giulia_obj.power_consumption_obj.consumption_per_RF_chain_W)
        consumption_per_MCPA_W_noisy = apply_relative_noise(self.giulia_obj.power_consumption_obj.consumption_per_MCPA_W)
        efficiency_of_MCPA_noisy = apply_relative_noise(self.giulia_obj.power_consumption_obj.efficiency_of_MCPA)

        # Create the radio unit parameters dictionary with noisy values
        radio_unit_parameters_d__noisy = {
                                          # Static Power P0 with multiplicative relative noise
                                          "P_0": P_0_noisy,

                                          # Continuous parameters use multiplicative relative noise
                                          "consumption_per_RF_chain_W": np.maximum(consumption_per_RF_chain_W_noisy, 0.0),
                                          "consumption_per_MCPA_W": np.maximum(consumption_per_MCPA_W_noisy, 0.0),

                                          # Efficiency must remain in the interval [0, 1]
                                          "efficiency_of_MCPA": np.clip(efficiency_of_MCPA_noisy, 0.0, 1.0),
                                    }
        
        # Update in Giulia
        for key, values in radio_unit_parameters_d__noisy.items():
            # check if exist in Giulia
            if not hasattr(self.giulia_obj.power_consumption_obj, key):
                raise AttributeError(f"Giulia's power_consumption_obj does not have the attribute '{key}'.")
            # set the attribute in Giulia's power consumption object
            setattr(self.giulia_obj.power_consumption_obj, key, values)


        # Optionally bar-plot the original vs updated (noisy) radio unit parameters.
        if kwargs.get("plot_ru_noisy_param", False):
            # Build the x-axis identity (ID + type) shared by both dictionaries.
            radio_unit_ids = np.arange(n_ru, dtype=int)
            radio_unit_types = np.asarray(
                self.giulia_obj.power_consumption_obj.radio_unit_types[
                    self.giulia_obj.power_consumption_obj.indices_unique_radio_units_IDs
                ]
            )

            # Compare only the parameters that were actually modified, so the
            # original and updated dictionaries share exactly the same keys.
            basic_parameters = {"ID": radio_unit_ids, "type": radio_unit_types}
            noisy_parameters = {"ID": radio_unit_ids, "type": radio_unit_types}
            for key in radio_unit_parameters_d__noisy:
                basic_parameters[key] = radio_unit_parameters_d[key]
                noisy_parameters[key] = radio_unit_parameters_d__noisy[key]

            self.plot_ru_basic_vs_noisy(basic_parameters, noisy_parameters)

        return


    def plot_ru_basic_vs_noisy(self, basic_parameters: dict, noisy_parameters: dict) -> None:
        """Bar-plot each radio unit parameter as original vs updated (noisy) values.

        Mirrors the ``plot_basic_vs_noisy`` helper from
        ``giulia_4G5G_multilayer.ipynb``: for every parameter shared by the two
        dictionaries it draws a grouped bar chart comparing the original values
        against the ones produced by :meth:`set_giulia_ru_random_parameters`.

        Both dictionaries must expose an ``"ID"`` and a ``"type"`` entry (used for
        the x-axis labels) plus one entry per radio-unit parameter. Plain numeric
        arrays are plotted directly; the dict-valued ``P_0`` parameter yields one
        chart per power state.

        Parameters
        ----------
        basic_parameters: dict
            Original radio-unit parameters (before noise).
        noisy_parameters: dict
            Updated radio-unit parameters (after noise).
        """
        ids = np.asarray(basic_parameters["ID"])
        radio_types = np.asarray(basic_parameters["type"])

        if not np.array_equal(ids, np.asarray(noisy_parameters["ID"])):
            raise ValueError("The basic and noisy dictionaries must have matching IDs.")

        x = np.arange(len(ids))
        bar_width = 0.4

        # Per-radio-unit x tick labels: "ID <id>\n<type>".
        x_labels = [
            f"ID {radio_id}\n{radio_type}"
            for radio_id, radio_type in zip(ids, radio_types)
        ]

        # "ID" and "type" identify the radio units; they are not plotted.
        excluded_keys = {"ID", "type"}

        for parameter_name, basic_entry in basic_parameters.items():

            if parameter_name in excluded_keys:
                continue

            if parameter_name not in noisy_parameters:
                warning(f"Skipping {parameter_name!r}: missing from noisy dictionary.")
                continue

            basic_values = np.asarray(basic_entry)
            noisy_values = np.asarray(noisy_parameters[parameter_name])

            if len(basic_values) != len(ids) or len(noisy_values) != len(ids):
                warning(
                    f"Skipping {parameter_name!r}: "
                    "array length does not match the number of radio units."
                )
                continue

            # Case 1: array containing dictionaries, for example P_0 (one bar
            # chart per power state).
            if all(isinstance(value, dict) for value in basic_values):

                if set(basic_values[0].keys()) != set(noisy_values[0].keys()):
                    raise ValueError(f"Dictionary keys do not match for {parameter_name!r}.")

                for state in basic_values[0].keys():
                    basic_state_values = np.array([value[state] for value in basic_values], dtype=float)
                    noisy_state_values = np.array([value[state] for value in noisy_values], dtype=float)
                    self._draw_ru_basic_vs_noisy_bars(
                        x, x_labels, bar_width,
                        basic_state_values, noisy_state_values,
                        title=f"{parameter_name} — {state}: original vs updated",
                        ylabel=f"{parameter_name} [{state}]",
                    )

            # Case 2: plain numeric NumPy array.
            elif np.issubdtype(basic_values.dtype, np.number) and np.issubdtype(noisy_values.dtype, np.number):
                readable_name = parameter_name.replace("_", " ")
                self._draw_ru_basic_vs_noisy_bars(
                    x, x_labels, bar_width,
                    basic_values, noisy_values,
                    title=f"{readable_name}: original vs updated",
                    ylabel=readable_name,
                )

            else:
                warning(f"Skipping {parameter_name!r}: unsupported dtype {basic_values.dtype}.")

        return


    def _draw_ru_basic_vs_noisy_bars(self, x, x_labels, bar_width,
                                     basic_values, noisy_values,
                                     title: str, ylabel: str) -> None:
        """Draw a grouped (original vs updated) bar chart for one RU parameter,
        then a separate figure with its value distribution across radio units.

        The distribution figure uses a KDE (:func:`seaborn.kdeplot`) when the
        values spread out; a parameter that is fixed across every radio unit
        would produce a degenerate KDE, so it is shown as a single vertical line
        instead.
        """
        fig, ax = plt.subplots(figsize=(max(9, len(x) * 1.1), 7))

        basic_bars = ax.bar(x - bar_width / 2, basic_values, width=bar_width, label="Original")
        noisy_bars = ax.bar(x + bar_width / 2, noisy_values, width=bar_width, label="Updated")

        ax.set_title(title)
        ax.set_xlabel("Radio unit")
        ax.set_ylabel(ylabel)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax.bar_label(basic_bars, fmt="%.3g", padding=3, fontsize=8)
        ax.bar_label(noisy_bars, fmt="%.3g", padding=3, fontsize=8)

        fig.tight_layout()
        plt.show()

        # Separate figure: value distribution across radio units, reusing the bar colours.
        self._draw_ru_value_distribution(
            {"Original": (basic_values, basic_bars[0].get_facecolor()),
             "Updated": (noisy_values, noisy_bars[0].get_facecolor())},
            xlabel=ylabel,
            title=f"{title} — distribution across radio units",
        )
        return


    def _draw_ru_value_distribution(self, series_dict: dict, xlabel: str, title: str) -> None:
        """Plot, in its own figure, the distribution of one RU parameter for each series.

        Parameters
        ----------
        series_dict: dict
            Maps a series label (e.g. ``"Original"``/``"Updated"``) to a
            ``(values, colour)`` tuple. A series whose values are identical
            across all radio units is drawn as a vertical line (a KDE would be
            degenerate); otherwise a filled KDE is drawn.
        xlabel: str
            Label for the value axis (matches the bar chart's y-axis).
        title: str
            Title for the distribution figure.
        """
        fig, ax = plt.subplots(figsize=(9, 6))

        for label, (values, color) in series_dict.items():
            values = np.asarray(values, dtype=float)

            # A parameter that is fixed across every radio unit has no spread:
            # mark it with a vertical line rather than a degenerate KDE.
            if values.size < 2 or np.allclose(values, values.flat[0]):
                ax.axvline(
                    values.flat[0], color=color, linestyle="--", linewidth=2,
                    label=f"{label} (fixed = {values.flat[0]:.3g})",
                )
            else:
                # Let seaborn compute the KDE, then normalise its peak to 1 so
                # series with very different spreads stay comparable in height
                # and both remain visible. (fill is added manually on the
                # normalised curve to keep it consistent with the line.)
                sns.kdeplot(x=values, ax=ax, color=color, label=label)
                curve_x, curve_y = ax.lines[-1].get_data()
                peak = curve_y.max() # type: ignore
                if peak > 0:
                    curve_y = curve_y / peak
                    ax.lines[-1].set_ydata(curve_y)
                ax.fill_between(curve_x, curve_y, color=color, alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized density (peak = 1)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()
        return
    

    def make_ue_contour_field(self, seed=11):
            
        """Build a smooth 2D scalar field made of random Gaussian hills.

        Returns
        -------
        X, Y : 2D coordinate arrays (meshgrid, "xy" indexing), spaced `step` m apart.
        Z    : 2D field values, rescaled to [z_low, z_high].
        """

        # Derive the field extent from Giulia's UE-playground scenario box, so the
        # field covers exactly the (x, y) area where UEs can be placed. This box is
        # populated by ue_playground_deployment_obj.process() during Giulia init.
        ue_playground_obj = self.giulia_obj.ue_playground_deployment_obj
        if not isinstance(ue_playground_obj, Site_uePlaygroundDeployment):
            raise TypeError("Giulia's ue_playground_deployment_obj is not a valid Site_uePlaygroundDeployment instance.")
        if ue_playground_obj.scenario_x_side_length_m is None or ue_playground_obj.scenario_y_side_length_m is None:
            raise ValueError("UE playground scenario side lengths are not set; ensure Giulia is initialised before building the field.")

        if not isinstance(ue_playground_obj.scenario_x_side_length_m, (int, float)): raise TypeError(
               f"scenario_x_side_length_m must be an int or float in {ue_playground_obj.__class__.__name__}")
        if not isinstance(ue_playground_obj.scenario_y_side_length_m, (int, float)): raise TypeError(
               f"scenario_y_side_length_m must be an int or float in {ue_playground_obj.__class__.__name__}")

        # Scenario box: centred on layout_centre_m, spanning +/- side/2 on each axis.
        x_centre, y_centre = ue_playground_obj.layout_centre_m[0], ue_playground_obj.layout_centre_m[1]
        half_x = float(ue_playground_obj.scenario_x_side_length_m) / 2
        half_y = float(ue_playground_obj.scenario_y_side_length_m) / 2
        x_min, x_max = x_centre - half_x, x_centre + half_x
        y_min, y_max = y_centre - half_y, y_centre + half_y

        step = 10          # grid spacing in meters (one sample every `step` m)
        n_peaks = 10
        smoothness = 275   # hill width (Gaussian sigma) in meters
        z_low = 0.25        # lower bound of the output range
        z_high = 1.75       # upper bound of the output range

        rng = np.random.default_rng(seed)

        # Number of samples so that consecutive points are exactly `step` m apart
        # and both endpoints are included.
        n_x = int(round((x_max - x_min) / step)) + 1
        n_y = int(round((y_max - y_min) / step)) + 1
        x = np.linspace(x_min, x_max, n_x)
        y = np.linspace(y_min, y_max, n_y)
        X, Y = np.meshgrid(x, y)  # shape (n_y, n_x)

        # Start from an empty field and drop random spikes onto it.
        Z = np.zeros((n_y, n_x))
        for _ in range(n_peaks):
            if _ == 0:
                i = n_y//2          # row index -> y
                j = n_x//2          # col index -> x
                Z[i, j] += rng.uniform(0.5, 1.0)  # += so overlapping spikes add up
            else:
                i = rng.integers(0, n_y)          # row index -> y
                j = rng.integers(0, n_x)          # col index -> x
                Z[i, j] += rng.uniform(0.5, 1.0)  # += so overlapping spikes add up

        # Blur the spikes into smooth hills. sigma is given in meters, so convert
        # it to pixels; this keeps the hill width independent of `step`.
        Z = gaussian_filter(Z, sigma=smoothness / step)

        # Normalize to [0, 1] (guarding against a flat field), then rescale to
        # the requested [z_low, z_high] range.
        z_range = Z.max() - Z.min()
        if z_range > 0:
            Z = (Z - Z.min()) / z_range
        Z = z_low + (z_high - z_low) * Z # type: ignore

        return X, Y, Z
    
    def ue_contour_field_lookup(self, X=None, Y=None, Z=None, method="linear"):
        """Return a callable f((x, y)) -> Z that interpolates the UE contour field.

        If X, Y and Z are omitted, the field cached in ``self.ue_countour_field_xyz``
        (built by :meth:`make_ue_contour_field`) is used.

        The returned callable accepts either a single point, where x and y are
        scalars (returns a float), or many points at once, where x and y are
        equal-length lists / arrays (returns a 1D array); a scalar is broadcast
        against an array, e.g.:

            z_at((0, 0))               -> float
            z_at(([0, 100], [0, 100])) -> array([z(0,0), z(100,100)])
            z_at((0, [0, 100]))        -> array([z(0,0), z(0,100)])

        Points outside the grid are clipped to the domain, so the lookup
        never raises on boundary/out-of-range queries.
        """
        # Fall back to the cached field when no explicit grid is passed.
        if X is None or Y is None or Z is None:
            X, Y, Z = self.ue_countour_field_xyz

        y_axis = Y[:, 0]
        x_axis = X[0, :]
        interpolator = RegularGridInterpolator(
            (y_axis, x_axis),  # Z is indexed as Z[y, x]
            Z,
            method=method,
            bounds_error=False,
            fill_value=float("nan"),
        )

        x_lo, x_hi = x_axis.min(), x_axis.max()
        y_lo, y_hi = y_axis.min(), y_axis.max()

        def z_at(point):
            x, y = point
            # Remember whether the caller passed a single (scalar, scalar) point,
            # so we can return a plain float in that case.
            scalar = np.ndim(x) == 0 and np.ndim(y) == 0
            x = np.clip(x, x_lo, x_hi)
            y = np.clip(y, y_lo, y_hi)
            # Broadcast to a common shape, then stack into (..., 2) query points
            # in (y, x) index order. Handles scalars, lists and arrays alike.
            y, x = np.broadcast_arrays(y, x)
            z = np.atleast_1d(interpolator(np.stack([y, x], axis=-1)))
            return float(z[0]) if scalar else z  # scalar in -> scalar out

        return z_at

    def plot_ue_contour_field(self, ue_countour_field_xyz=None, n_levels=11,
                              cmap="viridis", title="UE contour field", filled: bool = True):
        """Visualise the UE contour field as a filled contour map.

        Mirrors the plot in ``contour_levels_0_1.ipynb``: filled contours with
        overlaid labelled iso-lines and a colour bar.

        Parameters
        ----------
        ue_countour_field_xyz : tuple, optional
            (X, Y, Z) grid to plot. Defaults to the field cached in
            ``self.ue_countour_field_xyz`` (built by :meth:`make_ue_contour_field`).
        n_levels : int
            Number of contour levels spanning the field's value range.
        cmap : str
            Matplotlib colormap for the filled contours.
        title : str
            Axes title.
        filled : bool
            Whether to fill the contours with colour. If False, only the contour lines are drawn.

        Returns
        -------
        (fig, ax) : the created Matplotlib figure and axes.
        """
        # Fall back to the cached field when no grid is passed.
        if ue_countour_field_xyz is None:
            ue_countour_field_xyz = self.ue_countour_field_xyz
        X, Y, Z = ue_countour_field_xyz

        # Contour levels spanning the actual field range.
        levels = np.linspace(Z.min(), Z.max(), n_levels)

        fig, ax = plt.subplots(figsize=(8, 7))

        if filled:
            filled_ax = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            fig.colorbar(filled_ax, ax=ax, label="Value") # pyright: ignore[reportArgumentType]
        lines = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.7)
        ax.clabel(lines, inline=True, fontsize=8)


        ax.set_xlabel("x axis (m)")
        ax.set_ylabel("y axis (m)")
        ax.set_title(title)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect("equal", adjustable="box")  # keep limits, unlike axis("equal")
        ax.grid(True, alpha=0.3)

        # plt.show()
        return fig, ax
    

    def apply_ue_contouring_map(self, ue_per_UEsite_cell: np.ndarray) -> np.ndarray:
        """Apply the UE contouring map to the given UE density array.

        Parameters
        ----------
        ue_per_UEsite_cell : np.ndarray
            Array of UEs per UE site cell before contouring.

        Returns
        -------
        np.ndarray
            Array of UEs per UE site cell after applying the contouring map.
        """
        if getattr(self, "ue_countour_field_xyz", None) is None:
            # warning("UE contour field has not been generated. Call make_ue_contour_field() first.")
            return ue_per_UEsite_cell  # Return the original array if contouring is not applied.    

        X, Y, Z = self.ue_countour_field_xyz

        # Get the positions of the UE site cells from Giulia's deployment DataFrame.
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj, Site_uePlaygroundDeployment):
            raise TypeError("Giulia's ue_playground_deployment_obj is not a valid Site_uePlaygroundDeployment instance.")

        ue_playground_obj = self.giulia_obj.ue_playground_deployment_obj  # local alias for readability

        if not isinstance(ue_playground_obj.ref_cell_site_positions_m, np.ndarray):
            raise ValueError("Giulia's ue_playground_deployment_obj does not contain valid reference cell site positions.")

        # Read the cell site and the bearing of every UE-site cell off the playground maps rather
        # than assuming a full sectorization: a site whose sectors are not all present (a
        # downsampled df_ep) has fewer UE-site cells than sectors x cell sites, and the two maps
        # are the only description of which sectors actually carry UEs.
        number_of_UEsiteCells = len(ue_per_UEsite_cell)
        missing_UEsiteCells = [UEsiteCell for UEsiteCell in range(number_of_UEsiteCells)
                               if UEsiteCell not in ue_playground_obj.map_cell_to_site
                               or UEsiteCell not in ue_playground_obj.map_cell_to_bearing]
        if missing_UEsiteCells:
            raise ValueError(
                f"ue_per_UEsite_cell holds {number_of_UEsiteCells} UE-site cells but the UE playground maps "
                f"do not cover {len(missing_UEsiteCells)} of them (first: {missing_UEsiteCells[0]}).")

        site_index_per_UEsiteCell = np.array(
            [ue_playground_obj.map_cell_to_site[UEsiteCell] for UEsiteCell in range(number_of_UEsiteCells)], dtype=int)
        bearing_cell = np.array(
            [ue_playground_obj.map_cell_to_bearing[UEsiteCell] for UEsiteCell in range(number_of_UEsiteCells)], dtype=float)

        # Cell site position of every UE-site cell
        x_position_cell = ue_playground_obj.ref_cell_site_positions_m[site_index_per_UEsiteCell, 0]
        y_position_cell = ue_playground_obj.ref_cell_site_positions_m[site_index_per_UEsiteCell, 1]

        # Compute the actual positions of the UE site cells based on their bearings and a fixed distance (50 meters) from the site center.
        x_position_cell = x_position_cell + 100 * np.cos(np.radians(bearing_cell))
        y_position_cell = y_position_cell + 100 * np.sin(np.radians(bearing_cell))


        # Create an interpolator for the contour field.
        z_at = self.ue_contour_field_lookup(X, Y, Z)

        # Apply the contouring map to each UE site cell based on its position.
        contour_factors = z_at((x_position_cell, y_position_cell))
        
        # Adjust the UE density by the contour factors.
        adjusted_ue_per_UEsite_cell = ue_per_UEsite_cell * contour_factors

        return np.round(adjusted_ue_per_UEsite_cell).astype(int)



    # Sync Dedalo's radio status view with Giulia runtime data.
    def set_dedalo_radio_status_from_giulia_runTime(self) -> None:
        """Capture the most recent radio unit status from Giulia.

        The method supports two scenarios: before Giulia runtime (read from
        deployment data) or after runtime (read from BaseStation objects).

        Raises
        ------
        TypeError
            If Giulia exposes unexpected container types for deployment objects.
        ValueError
            If the deployment DataFrame is missing or invalid.
        """
        # Very first ever iteration before Giulia confif and runtime
        if isinstance(self.giulia_obj.base_stations_obj, list):

            if not isinstance(self.giulia_obj.network_deployment_obj, Network):
                raise TypeError("The network_deployment_obj should be a Network object, not a list.")
            if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
                raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
            df_ep = self.giulia_obj.network_deployment_obj.df_ep
            self.giulia_radio_status_dict = dict(zip(df_ep["name"], df_ep["radio_unit_initial_status"]))


        # After Giulia first configuration and runtime
        elif isinstance(self.giulia_obj.base_stations_obj, BaseStations):
            # if first episode, reset all to active
            if self.episode_index == 0:
                self.giulia_radio_status_dict =\
                    {bs_obj.cell_name: "active" for bs_obj in self.giulia_obj.base_stations_obj.base_stations}
                
            # else, get from runtime
            else:
                self.giulia_radio_status_dict =\
                    {bs_obj.cell_name: bs_obj.rrc_layer.radio_unit_status for bs_obj in self.giulia_obj.base_stations_obj.base_stations}

        else:
            raise TypeError("Wrong type for giulia_obj.base_stations_obj. It should be a BaseStations object.")
        
        return
    
    # Write Dedalo's radio status snapshot back into Giulia's deployment frame.
    def set_giulia_df_ep_radio_status(self) -> None:
        """Write cached radio unit status back into Giulia's deployment frame.

        Raises
        ------
        TypeError
            If the Giulia deployment object is not a Network instance.
        ValueError
            If the deployment DataFrame is missing or invalid.
        """
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        for cell_name, status in self.giulia_radio_status_dict.items():
            self.giulia_obj.network_deployment_obj.df_ep.loc[self.giulia_obj.network_deployment_obj.df_ep['name'] == cell_name, 'radio_unit_initial_status'] = status

        return

    def _setGiulia_ue_traffic_generation_model(self, model: str='rate_requirement') -> None:
        """
        The following method set in giulia UE dataframe engeneer parmaeter df_ep the traffic generation model.
        """
        self.set_giulia_ue_df_ep(name_ue_list=self.get_ue_names(attribute=None),
                                 name_attribute='traffic_generation_model',
                                 value_list=model)
        return

    def _get_target_rate_per_ue_Mbps(self, mu_lin: float = 1.0,
                                     sigma_lin: float = 1.0,
                                     n_sample: int = 1,
                                     seed: int|None = None) -> np.ndarray:
        
        # Set Seed
        seed = 42 if seed is None else seed
        rng: np.random.RandomState = np.random.RandomState(seed)
        """Generate a target rate for a UE based on a log-normal distribution."""

        x = rng.lognormal(mean=np.log(mu_lin), sigma=np.log(sigma_lin), size=n_sample).astype(np.float32)
        
        # Max value for the target rate in Mbps
        p = 0.95
        x = np.clip(x, 0, 
                    np.exp( np.log(mu_lin) + np.log(sigma_lin) * norm.ppf(p) ) )

        return x
    
    def setGiulia_ue_target_rate(self, ue_rate_requirements_Mbps: list) -> None:
        """
        The following method set in giulia UE dataframe engeneer parmaeter df_ep the target rate for each UE.
        """
        self.set_giulia_ue_df_ep(name_ue_list=self.get_ue_names(attribute=None),
                                 name_attribute='ue_target_rate_Mbps',
                                 value_list=ue_rate_requirements_Mbps)
        
        return 


    def plot_scenario_state(self) -> None:
        """
        The following method plots the scenario state, including the contour field, cell sites, and UEs. The contour field is generated using the make_ue_contour_field() method, 
        and the cell sites and UEs are overlaid on top of it.
        
        # first, it gets the fig and ax from the dedalo playground with contour levels, as returned by dedalo.make_ue_contour_field() 
        # then, it scatter the cell sites on the contour plot. Out from the cell sites small lines points towards the different boresight of each RU.
        # the color of the arrows indicates the state of the co-located RUs. 
        # Black, inactive
        # Red, the 4G cell is active, the 5G cell is inactive
        # Blue, the 4G cell is inactive, the 5G cell is active
        # Green, both 4G and 5G cells are active
        # first 57 cells are 4G, the next 57 cells are 5G, the last cell is the HAPS hypercell.
        # the state of the cells shall be inferred from giulia, so that it avoids mismatch with env action choices.
        # then, it scatters the UEs on the contour plot, with color indicating the associated cell. 
        # Red if the UE is associated to a 4G cell, 
        # Blue if the UE is associated to a 5G cell,
        # Black if the UE is associated with the HAPS

        """
        if not isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment):
            raise TypeError("Giulia's ue_deployment_obj is not a valid UE_Deployment instance.")
        if not isinstance(self.giulia_obj.ue_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Giulia's ue_deployment_obj does not contain a valid DataFrame.")

        if not isinstance(self.giulia_obj.ue_playground_deployment_obj, Site_uePlaygroundDeployment):
            raise TypeError("Giulia's ue_playground_deployment_obj is not a valid Site_uePlaygroundDeployment instance.")
        if self.giulia_obj.ue_playground_deployment_obj.scenario_x_side_length_m is None or self.giulia_obj.ue_playground_deployment_obj.scenario_y_side_length_m is None:
            raise ValueError("UE playground scenario side lengths are not set; ensure Giulia is initialised before building the field.")
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj.scenario_x_side_length_m, (int, float)): raise TypeError(
               f"scenario_x_side_length_m must be an int or float in {self.giulia_obj.ue_playground_deployment_obj.__class__.__name__}")
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj.scenario_y_side_length_m, (int, float)): raise TypeError(
               f"scenario_y_side_length_m must be an int or float in {self.giulia_obj.ue_playground_deployment_obj.__class__.__name__}")


        UE_color_map =   {"4G": "red", "5G": "blue", "HAPS": "black"}
        site_color_map = {"4G": "red", "5G": "blue", "HAPS": "black", "both": "green", "inactive": "grey"}

        # plot the countour field, with filled=False to only show the contour lines to ease representation of the cell sites and UEs on top of it.
        fig, ax = self.plot_ue_contour_field(filled=False)
        
        #### PLOT UE
        # get UE positions from Giulia and scatter them on the contour plot, with color indicating the associated cell.
        UE_positions_x = np.array(self.giulia_obj.ue_deployment_obj.df_ep['position_x_m'])
        UE_positions_y = np.array(self.giulia_obj.ue_deployment_obj.df_ep['position_y_m'])

        UE_serving_cell_ID = self.giulia_obj.best_serving_cell_ID_per_ue_based_on_SSB_obj.best_serving_cell_ID_per_ue
        UE_serving_cell_type = np.where(UE_serving_cell_ID < 57, "4G", np.where(UE_serving_cell_ID < 2*57, "5G", "HAPS"))

        UE_colors = [UE_color_map[ue_type] for ue_type in UE_serving_cell_type]
        ax.scatter(UE_positions_x, UE_positions_y, c=UE_colors, marker='o')

        #### PLOT CELL SITES
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj.ref_cell_site_positions_m, np.ndarray):
            raise ValueError("ref_cell_site_positions_m is not set; run _apply_ref_layout_from_df_ep first.")  # guard: layout helper must have run
        
        # get the site positions from Giulia and scatter them on the contour plot, with arrows indicating the boresight of each RU. The color of the arrows indicates the state of the co-located RUs.
        site_positions_x = self.giulia_obj.ue_playground_deployment_obj.ref_cell_site_positions_m[:,1]
        site_positions_y = self.giulia_obj.ue_playground_deployment_obj.ref_cell_site_positions_m[:,0]
        
        # each site has 3 sector
        ref_number_of_sectors_per_site = self.giulia_obj.ue_playground_deployment_obj.ref_number_of_sectors_per_site
        # Get the horizontal antenna bearing angles for each UE site cell from Giulia's deployment object.
        ref_antenna_config_hor_alpha_mec_bearing_deg = self.giulia_obj.ue_playground_deployment_obj.ref_antenna_config_hor_alpha_mec_bearing_deg

        # Check
        if ref_number_of_sectors_per_site is None:
            raise ValueError("ref_number_of_sectors_per_site is not set; run _apply_ref_layout_from_df_ep first.")
        if ref_antenna_config_hor_alpha_mec_bearing_deg is None:
            raise ValueError("ref_antenna_config_hor_alpha_mec_bearing_deg is not set; run _apply_ref_layout_from_df_ep first.")
        
        # Repeat the site cell positions for each sector 
        x_position_cell = np.repeat(site_positions_x, ref_number_of_sectors_per_site)
        y_position_cell = np.repeat(site_positions_y, ref_number_of_sectors_per_site)

        # Repeat the bearing angles for each sector
        bearing_cell = np.tile(ref_antenna_config_hor_alpha_mec_bearing_deg, len(site_positions_x))

        # Calculate off-set for the arrows based on the bearing angles to indicate the direction of the boresight of each RU.
        arrow_length = 150  # Length of the arrows
        arrow_dx = arrow_length * np.cos(np.radians(bearing_cell))
        arrow_dy = arrow_length * np.sin(np.radians(bearing_cell))

        # get binary radio unit status for all cells from Giulia, where 1 indicates active and 0 indicates carrier shutdown
        # determine the color of the arrows based on the state of the co-located RUs. 
        # The couple 0 and 57 refers to the 4G cell and 5G cell co-located, similarly 1 and 58 to the second couple, etc...
        if not isinstance(self.giulia_obj.base_stations_obj, BaseStations):
            raise TypeError("Giulia's base_stations_obj is not a valid BaseStations instance.")
        
        radio_unit_status = np.array([bs_obj.rrc_layer.radio_unit_status for bs_obj in self.giulia_obj.base_stations_obj.base_stations])
        radio_unit_status = np.where(radio_unit_status == "active", 1, 0)

        tn_radio_unit_status = radio_unit_status[:-1]  # Exclude the last element which corresponds to the HAPS hypercell
        arrow_colors = []
        for i in range(len(radio_unit_status)//2):
            if tn_radio_unit_status[i] == 1 and tn_radio_unit_status[i+57] == 0:
                arrow_colors.append(site_color_map["4G"])  # 4G active, 5G inactive
            elif tn_radio_unit_status[i] == 0 and tn_radio_unit_status[i+57] == 1:
                arrow_colors.append(site_color_map["5G"])  # 4G inactive, 5G active
            elif tn_radio_unit_status[i] == 1 and tn_radio_unit_status[i+57] == 1:
                arrow_colors.append(site_color_map["both"])  # both active
            else:
                arrow_colors.append(site_color_map["inactive"])  # both inactive

        ax.quiver(x_position_cell, y_position_cell, arrow_dx, arrow_dy, angles='xy', scale_units='xy', scale=1, color=arrow_colors)

        #create a legend for the arrows indicating the state of the co-located RUs 
        legend_elements = [
            Line2D([0], [0], color=site_color_map["4G"], lw=2, label='4G ON, 5G OFF'),
            Line2D([0], [0], color=site_color_map["5G"], lw=2, label='4G OFF, 5G ON'),
            Line2D([0], [0], color=site_color_map["both"], lw=2, label='both ON'),
            Line2D([0], [0], color=site_color_map["inactive"], lw=2, label='both OFF')
        ]

        fig.legend(handles=legend_elements, loc='lower center', ncol=4)

        # plt.show()
        # fig.savefig("test.png", dpi=300)
        del fig, ax
        plt.close()
        
        return  
    

    def process_static(self, episode_index: int = 0, **kwargs) -> None:
        """Process static episode-level information for Dedalo.

        Parameters
        ----------
        episode_index: int, optional
            Index of the episode being processed, by default 0.
        **kwargs:
            Additional episode-level information, if needed.
        """
        
        self.episode_index = max(episode_index, 0)
        self.step_index = 0
        del episode_index

        # # Get the UE density for the current hour.
        value_ue_per_UEsite_cell_per_specific_hour = self._get_ue_per_UEsite_cell_per_specific_hour(self.episode_index + self.start_hour)

        # Apply contouring map to adjust the UE density based on spatial distribution.
        value_ue_per_UEsite_cell_per_specific_hour = self.apply_ue_contouring_map(value_ue_per_UEsite_cell_per_specific_hour)

        # Update number of UE per UE site cell in Giulia for the current hour.
        self.set_in_Giulia_ue_per_UEsite_cell(value_ue_per_UEsite_cell_per_specific_hour)

        # Sync the radio unit status between Dedalo and Giulia.
        self.set_dedalo_radio_status_from_giulia_runTime()
        # Set in giulia previous radio status to mantain consistency among episodes
        self.set_giulia_df_ep_radio_status()

        threshold_PRBusage_shutdown_value = kwargs.get("threshold_PRBusage_shutdown_value", 0.30)
        threshold_PRBusage_wakeup_value = kwargs.get("threshold_PRBusage_wakeup_value", 0.70)

        # Update the shutdown and wakeup thresholds.
        # Using parent method to set values in df_ep.
        coverage_cell_name_list: list = self.get_cell_names("cell_role", "coverage")
        capacity_cell_name_list: list = self.get_cell_names("cell_role", "capacity")
        coverage_capacity_cell_name_list: list = self.get_cell_names("cell_role", "coverage_capacity")

        # Add coverage_capacity cells to both lists
        coverage_cell_name_list.extend(coverage_capacity_cell_name_list)
        capacity_cell_name_list.extend(coverage_capacity_cell_name_list)
        coverage_cell_name_list = sorted(list(set(coverage_cell_name_list)))
        capacity_cell_name_list = sorted(list(set(capacity_cell_name_list)))

        assert len(coverage_cell_name_list) > 0 and len(capacity_cell_name_list) > 0, "No coverage or capacity cells found in the deployment."
        

        self.set_giulia_network_df_ep(name_cell_list=capacity_cell_name_list, 
                                      name_attribute="carrier_shutdown_currentCell_usage_threshold", 
                                      value_list=threshold_PRBusage_shutdown_value)
            
        self.set_giulia_network_df_ep(name_cell_list=coverage_cell_name_list,  
                                      name_attribute="carrier_wakeUp_pairedCell__currentCell_usage_threshold", 
                                      value_list=threshold_PRBusage_wakeup_value)
        
        # Set Traffic Rate Gen Stats
        self.traffic_rate_gen_stats: dict|None = kwargs.get("traffic_rate_gen_stats", {"mu_lin": 5.0, "sigma_lin": 4.5})


        # Set Rescheduling info
        if not kwargs.get("activate_per_step_channel_comp", False):
            self._set_rescheduling_info()

        return
    


    
    
    def process_realTime(self, step_index: int = 0, **kwargs) -> None:
        """Process real-time step-level information for Dedalo.

        Parameters
        ----------
        step_index: int, optional
            Index of the step being processed, by default 0.
        **kwargs:
            Additional step-level information, if needed.
        """
        
        # Get the current step index.
        self.step_index = step_index
        del step_index

        #### Fixed PRB per UE
        # Retrieve the rate requirement for UEs. Defaults to None.
        kost_prb_usage_per_ue = kwargs.get("kost_prb_usage_per_ue", None)

        if kost_prb_usage_per_ue is not None:
            assert isinstance(self.giulia_obj.base_stations_obj, BaseStations), "giulia_obj.base_stations_obj must be set"
            
            # Use setattr to dynamically set the attribute
            if hasattr(self.giulia_obj.base_stations_obj, 'fix_prb_req_per_ue'):
                setattr(self.giulia_obj.base_stations_obj, 'fix_prb_req_per_ue', kost_prb_usage_per_ue)
            else:
                raise AttributeError("giulia_obj.base_stations_obj must have the attribute 'fix_prb_req_per_ue'")


        #### Generate Traffic for UEs
        if self.traffic_rate_gen_stats is not None:
            # Set traffic generation model
            self._setGiulia_ue_traffic_generation_model(model='rate_requirement')

            # Get the target rate for each UE
            self.ue_rate_requirements_Mbps = self._get_target_rate_per_ue_Mbps(mu_lin=self.traffic_rate_gen_stats['mu_lin'],
                                                                              sigma_lin=self.traffic_rate_gen_stats['sigma_lin'],
                                                                              n_sample=self.giulia_number_of_ue,
                                                                              seed=self.episode_index)
            
            # Set the target rate for each UE in Giulia
            self.setGiulia_ue_target_rate(
                ue_rate_requirements_Mbps=self.ue_rate_requirements_Mbps.tolist()
            )

    
        return
    