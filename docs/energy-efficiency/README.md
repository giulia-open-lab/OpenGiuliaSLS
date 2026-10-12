# Energy Efficiency

Documentation for Giulia's **energy-efficiency** features — the mechanisms that reduce
radio access network (RAN) power consumption while keeping users served.

Start with the **Overview**, then read the topic pages in order.

| Page | What it covers |
|---|---|
| **[Overview](./overview.md)** | Orientation: the three things to know up front, what the features are, how they fit together, where they run, and how savings are measured. Acronym glossary. |
| **[RRM](./rrm.md)** | The radio-resource-management chain the feature lives inside: association, frequency-priority re-selection, scheduling, the `_prb_usage` load metric, intra-step event ordering, and exactly what happens to UEs when a carrier is switched off. |
| **[The RRC layer](./rrc-layer.md)** | The plumbing: the `RRC` and `Carrier_Shutdown` classes, the inter-cell mailbox and its name-matching defect, construction and scheduling. |
| **[Pairing](./pairing.md)** | Cell roles, the per-layer fallback adjacency model, the results cache, and the two failure modes of the pairing code. |
| **[Carrier shutdown](./carrier-shutdown.md)** | The algorithm: states, the shutdown and wake-up rules, stability, what it actually saves in watts, and the flags that keep it switched off. |

## Fast map of the code

| Concern | File | Key symbol(s) |
|---|---|---|
| Cell roles (coverage/capacity) | [`giulia/bs/bs_deployments.py`](../../giulia/bs/bs_deployments.py) | `_compute_coverage_and_capacity_frequencies`, `set_dfEp_coverage_capacity_cell_roles`, `set_dfEP_carrier_shutdown_attributes` |
| Pairing | [`giulia/rrc/cell_pairing.py`](../../giulia/rrc/cell_pairing.py) | `PairingCell` |
| RRC + carrier shutdown | [`giulia/rrc/rrcs.py`](../../giulia/rrc/rrcs.py) | `RRC`, `Carrier_Shutdown` |
| Association / re-association | [`giulia/rrc/cell_selections.py`](../../giulia/rrc/cell_selections.py), [`cell_re_selections.py`](../../giulia/rrc/cell_re_selections.py) | `Cell_Selection`, `Cell_Re_Selection` |
| Load metric | [`giulia/mac/schedulers.py`](../../giulia/mac/schedulers.py) | `BaseScheduler._prb_usage` |
| TX power on/off | [`giulia/bs/bs_tx_powers.py`](../../giulia/bs/bs_tx_powers.py) | `shutdown_singleBS_tx`, `set_initial_singleBS_tx_power_dBm` |
| Power consumption KPI | [`giulia/kpis/power_consumptions.py`](../../giulia/kpis/power_consumptions.py) | `Power_Consumption.calculate_power_consumption` |
| Wiring & scheduling | [`giulia/bs/base_stations.py`](../../giulia/bs/base_stations.py) | `initialize_eventDriven_RRC_layers`, `set_paired_cell` |
| Event loop | [`giulia/event_driven/eventDriven.py`](../../giulia/event_driven/eventDriven.py) | `EventDrivenSimulation.add_event`, `.run` |
| Scenario-specific overrides | [`dedalo/dedalo_EE.py`](../../dedalo/dedalo_EE.py) | `set_pairing_df_ep_as_per_terrestrial_haps_hierarchical`, `activate_haps_multiple_wakeup_logic` |

> ⚠️ Carrier shutdown is **disabled by default** — `radio_enable_carrier_shutdown` is
> `False` for every cell and nothing in `giulia/` sets it back. See
> [Carrier shutdown § 7.1](./carrier-shutdown.md#71-the-feature-ships-disabled) for how to
> enable it, and the rest of § 7 for the other caveats.
