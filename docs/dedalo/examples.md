# Dedalo — Worked examples

Runnable examples, in increasing order of realism. Every listing on this page was executed
against this tree; the outputs shown are the real ones.

**Environment.** Use the `giulia` Conda environment. All examples assume the repository root
is on `sys.path` (`dedalo/` is a namespace package with no `__init__.py`, so it is imported
from the root, not installed):

```python
import sys
sys.path.insert(1, "/path/to/Giulia")     # repository root
import matplotlib; matplotlib.use("Agg")  # headless: Dedalo's plot helpers import pyplot
```

---

## 1. Inspecting a deployment without running anything

The cheapest useful thing Dedalo does: bind to a fresh Giulia and look at the configurable
surface. `Dedalo` is abstract, so a throwaway concrete subclass is needed.

```python
from giulia.logger import set_log_level, parse_log_level
set_log_level(parse_log_level(3))                     # 3 = WARNING; levels run verbose→quiet

from examples.giulia_EV import get_giulia_class
from dedalo.dedalo import Dedalo

class Probe(Dedalo):
    """Read-only Dedalo: satisfies the ABC, changes nothing."""
    def process_static(self, episode_index=0, **kw): pass
    def process_realTime(self, step_index=0, **kw): pass

preset = "GiuliaStd"
Giulia = get_giulia_class(preset)
g = Giulia(preset, "3GPPTR38_901_5G", None, "uniform", None,
           "downlink", None, 0, False, 1, False, False)

d = Probe(g)

print(d.giulia_number_of_cells)          # 57
print(d.get_cell_names(None)[:3])        # ['cell_0', 'cell_1', 'cell_2']
print(d.get_cell_names("RAT", "NR"))     # every 5G cell
print(g.network_deployment_obj.df_ep.shape)   # (57, 96)
```

The positional arguments to `Giulia(...)` are, in order: `preset`,
`scenario_playground_model`, `ue_playground_model`, `ue_distribution`, `ue_mobility`,
`link_direction`, `wraparound`, `save_results`, `plots`, `number_of_episodes`, `regression`,
`enable_GPU`.

> Calling `d.get_ue_names(None)` here raises `TypeError` — the UEs do not exist until
> `g.configure()`. See [The API contract § 4](./api.md#4-what-is-legal-when).

---

## 2. A minimal Dedalo from scratch

A complete controller: switch a chosen set of cells off for the episode, run, and measure.
This is the smallest thing that is recognisably an agent's environment.

```python
import numpy as np
from dedalo.dedalo import Dedalo

class DedaloShutdown(Dedalo):
    """Switches a chosen set of cells off for the episode."""

    def __init__(self, giulia, off_cells=()):
        super().__init__(giulia)
        self.off_cells = list(off_cells)

    def process_static(self, episode_index=0, **kw):
        self.episode_index = episode_index
        all_cells = self.get_cell_names(None)
        status = ["carrier_shutdown" if c in self.off_cells else "active"
                  for c in all_cells]
        self.set_giulia_network_df_ep(all_cells, "radio_unit_initial_status", status)

    def process_realTime(self, step_index=0, **kw):
        self.step_index = step_index


def run(off_cells):
    g = Giulia(preset, "3GPPTR38_901_5G", None, "uniform", None,
               "downlink", None, 0, False, 1, False, False)
    d = DedaloShutdown(g, off_cells=off_cells)

    # ---- the canonical loop -------------------------------------------------
    d.process_static(episode_index=0)          # 1. episode setup, BEFORE configure
    g.configure(0, int(2e6))                   # 2. Giulia builds UEs + links
    for step in range(2):
        d.process_realTime(step)               # 3. step setup, BEFORE run
        g.run_simulation(step, int(1e6))       # 4. advance
    # -------------------------------------------------------------------------

    P  = np.asarray(g.performance_obj.r_e_total_network_power_consumption_kW)
    R  = np.asarray(g.performance_obj.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps)
    st = np.asarray(g.performance_obj.r_e_cell_radio_unit_statuses)
    return P, R, st

P0, R0, st0 = run([])
P1, R1, st1 = run([f"cell_{i}" for i in range(0, 57, 3)])   # every 3rd cell
```

Actual output:

```
BASELINE: all cells active
   r_e_cell_radio_unit_statuses shape (2, 57); shutdown count last step = 0
   total network power kW (per step) : [27.447, 27.447]
   mean UE throughput Mbps           : 166.090   (shape (2, 570))

ACTION: every 3rd cell shut down  (19 cells)
   r_e_cell_radio_unit_statuses shape (2, 57); shutdown count last step = 19
   total network power kW (per step) : [24.428, 24.428]
   mean UE throughput Mbps           : 122.523   (shape (2, 570))

DELTA  power 27.447 -> 24.428 kW (-11.0%)   throughput 166.090 -> 122.523 Mbps (-26.2%)
```

Points worth noticing:

- `radio_unit_initial_status` needs **no re-process**. It is consumed by `base_stations_obj`,
  which `configure()` rebuilds every episode — verified by running it both ways and getting
  identical power. See [the re-process rule](./api.md#5-the-re-process-rule).
- `r_e_cell_radio_unit_statuses` confirms the write reached the simulation: 19 cells report
  `carrier_shutdown`.
- Power is flat across the two steps because nothing in this controller varies per step. A
  controller that reacts to load would move it.
- **The read side here is the legacy `r_e_*` surface**, kept because that is what was actually
  executed to produce the numbers above and what the agents in this tree use. Reading results
  is `Saveable`'s job, not Dedalo's — the declared equivalents are
  `BaseStations.cell_radio_unit_statuses` and
  `Rate_based_on_eff_CSI_RS_SINR.ue_throughput_Mbps` via `Saveable.get`, with the caveat in
  [KPIs § 1](./kpis.md#known-gap-quantities-the-legacy-surface-has-and-saveable-does-not) that
  the network *total* power has no declared counterpart yet.

---

## 3. A write that needs repair and re-process

The contrast case. `dl_bandwidth_MHz` is consumed by `time_frequency_resource_obj`, built in
`Giulia.__init__` and **never rebuilt by `configure()`** — so the plain write is not enough.

### The right way

`repair=True` is the default, so the derived `dl_PRBs_available` is recomputed for you. What
remains is rebuilding the carrier object, which Dedalo leaves to the caller:

```python
class DedaloBandwidth(Dedalo):
    def process_static(self, episode_index=0, **kw):
        # 1. WRITE  (+ 2. REPAIR the derived column — the default)
        self.set_giulia_network_df_ep(self.get_cell_names(None), "dl_bandwidth_MHz", 50)

        # 3. RE-PROCESS the object that cached it
        self.giulia_obj.time_frequency_resource_obj.process()

    def process_realTime(self, step_index=0, **kw): pass
```

Reports **84.363 Mbps** — correct for 65 PRB at 50 MHz / 60 kHz SCS.

### What the other combinations do

All four, measured on this tree:

| Write | Re-process? | `df_ep` PRBs | Mean UE throughput |
|---|:---:|---:|---|
| **default** (`repair=True`) | ✅ | 65 | **84.363 Mbps** — correct |
| **default** (`repair=True`) | ❌ | 65 | 💥 `RuntimeError: size of tensor a (65) must match tensor b (135)` |
| `repair=False` | ✅ | 135 | 175.457 Mbps — silently wrong, **2.08×** |
| `repair=False` | ❌ | 135 | 175.457 Mbps — silently wrong, **2.08×** |

The default path can no longer produce a silently-wrong number: it either works or it fails
loudly. Reaching a wrong-but-quiet result now requires passing `repair=False` on purpose.

Full discussion in [`df_ep` consistency § 3](./df-ep-consistency.md#3-the-measured-cost-of-getting-it-wrong).

`Dedalo_UnRL.set_BW_scs_prb_numerology` is the production version — the same three steps, plus
the uplink, a per-RAT split, and one `_update_giulia_object_static(skip_pairing=True)` covering
the whole batch of writes.

---

## 4. Guarding with the checklist

`repair=True` handles the columns with a registered updater, but two columns
(`SSB_number_of_beams`, `CSI_RS_number_of_beams`) have none, and direct `df_ep` assignment
bypasses the API entirely. Asserting is cheap — pure pandas arithmetic on 57 rows — so it is
worth doing rather than hoping.

```python
class DedaloGuarded(Dedalo):
    def process_static(self, episode_index=0, **kw):
        ...                                     # writes here

        self.assert_df_ep_consistent()          # raises ValueError naming the stale columns
```

The longhand, if you want the results rather than an exception:

```python
results = d.check_df_ep()                        # {check_name: (passed, message)}
failed  = [k for k, (ok, _) in results.items() if not ok]

d.check_df_ep(verbose=True)                      # coloured report to the terminal
```

Demonstrated end to end — the legacy raw write, the guard catching it, and the default:

```
baseline cell_0:                   {'dl_bandwidth_MHz': 100, 'dl_subcarrier_spacing_kHz': 60, 'dl_PRBs_available': 135}
checklist:                         11/11 pass

>>> LEGACY raw write:  set_giulia_network_df_ep(..., repair=False)
  after write:                     {'dl_bandwidth_MHz': 50, 'dl_subcarrier_spacing_kHz': 60, 'dl_PRBs_available': 135}
  checklist:                       10/11 pass  FAILING: ['_check_dl_PRBs_available']

>>> assert_df_ep_consistent() now refuses to continue:
   ValueError: df_ep is inconsistent — a derived column does not match its sources: ['_check_dl_PRBs_available']

>>> DEFAULT write (repair=True) on the same column:
  after write:                     {'dl_bandwidth_MHz': 50, 'dl_subcarrier_spacing_kHz': 60, 'dl_PRBs_available': 65}
  checklist:                       11/11 pass
   assert_df_ep_consistent(): no raise
```

The checklist also runs standalone against a pristine deployment, which is the fastest way to
sanity-check a scenario constructor:

```bash
python tests/network_df_ep_check_list.py
```

---

## 5. Example: swapping in a real network deployment

Everything so far writes *columns* of `df_ep`. This example does the deeper thing Dedalo can
do: **replace the whole deployment with a measured one**, then re-derive the Giulia state that
was built from the synthetic layout it came with.

> This is one example of the pattern, using a measured deployment of **153 cells (93 LTE + 60 NR)
> across 32 sites**, which resolve to **96 UE-site cells** — see below. The dataset itself is not
> part of this repository; any CSV carrying `df_ep`'s 96 columns loads the same way.

### Why it needs more than a write

Giulia builds its playground, site layout, UE playground and cell maps during
`Giulia.__init__`, from the *scenario* it was given. Loading a dataset invalidates all of it —
the new network has a different number of sites, different positions, a different sectorization.
Writing `df_ep` alone would leave Giulia's geometry describing the deployment you replaced.

So Dedalo re-derives that state. This is the one place where it reaches past `df_ep` into
Giulia's internals, and it is worth reading if you plan to extend it.

### The setup

```python
from dedalo.dedalo_UnRL import Dedalo_UnRL

# 1. Giulia still needs a REAL scenario to build against — the dataset replaces the
#    deployment moments later, but the propagation and playground models come from here.
g = Giulia(preset, "3GPPTR38_901_4G_5G_multilayer", None, "uniform", None,
           "downlink", None, 0, False, 1, False, False)

# 2. Pin the synthetic layout to a single tier, then re-derive everything built on it.
#    Done BEFORE Dedalo, because the dataset's UE playground is sized against whatever
#    layout Giulia is holding at that moment.
g.site_deployment_obj.hex_layout_tiers = 1
g.ue_playground_deployment_obj.hex_layout_tiers = 1
for name in ("ue_hotspot_deployment_obj", "network_deployment_obj", "pairing_cell_obj",
             "cell_antenna_array_structure_obj", "cell_SSB_conf_obj", "cell_CSI_RS_conf_obj",
             "power_consumption_obj", "cell_re_selection_conf_obj",
             "time_frequency_resource_obj", "mcs_resource_obj", "bler_vs_sinrs_lut_obj",
             "shadowing_map_cell_obj"):
    getattr(g, name).process()

# 3. Dedalo loads the dataset and realigns Giulia to it.
dedalo = Dedalo_UnRL(g, scenario_model="data_driven_vlc",
                     data_set_path="path/to/your_df_ep.csv",
                     start_hour=6)

# 4. Apply the numerology the study runs at. TERRESTRIAL cells are selected by RAT,
#    so this must precede anything that appends a non-terrestrial cell.
dedalo.set_BW_scs_prb_numerology(BW_MHz_5G=100, scs_kHz_5G=60)
```

From here the loop is the ordinary one — `process_static` → `configure` →
`process_realTime` → `run_simulation`.

### What Dedalo calls inside Giulia

`scenario_model="data_driven_vlc"` routes `__init__` through `_init_data_driven_vlc`, which
reads the CSV onto `network_deployment_obj.df_ep` and then repairs Giulia around it:

| Dedalo step | Giulia internals it drives | Why |
|---|---|---|
| `_update_giulia_NetworkObj_after_dataset_load` | `map_cellName_to_cellIndex` / `map_cellIndex_to_cellName`, `_compute_coverage_and_capacity_frequencies`, `set_dfEp_coverage_capacity_cell_roles`, `_init_to_None_cell_pairing_in_df_ep`, `set_dfEP_carrier_shutdown_attributes`, `basic_checks` | Name↔index maps, cell roles and carrier-shutdown attributes were derived from the old deployment |
| `_update_giulia_siteDeploymentObj_after_dataset_load` | `Site_siteDeployment.ref_cell_site_positions_m`, `ref_number_of_cell_sites`, `scenario_*_side_length_m`, `layout_centre_m`, `map_cell_to_site`, `map_cell_to_bearing` | The site grid is now the dataset's real positions, not a hexagon |
| `_update_giulia_uePlaygroundObj_after_dataset_load` | the same layout fields on `Site_uePlaygroundDeployment`, plus `Simulation_Config._set_ue_playground_deployment_obj` and `._set_number_of_ues_per_UEsite_cell` | UEs must drop over the real footprint, and the UE count per sector follows from it |
| `_update_giulia_object_static()` | `process()` on the whole `__init__`-built chain | Antennas, codebooks, power model and carriers were all built from the old table |

Two subtleties worth knowing if you write your own loader:

- **Site cells vs UE-site cells are counted differently.** A *site cell* is one `df_ep` row.
  A *UE-site cell* is a distinct (site, bearing) pair. On this dataset that is **153 rows but
  96 sectors**: the 4G and 5G carriers of a site share its sectors, and UEs must be dropped
  once per sector, not once per carrier. Counting rows would inflate the UE population by 60%.
  See `_update_siteCells_maps_from_df_ep` versus `_update_UEsiteCells_maps_from_df_ep`.
- **Sites are de-duplicated on the coordinate *pair*.** `df_ep` repeats a site's `(x, y)` once
  per cell; `_apply_ref_layout_from_df_ep` calls `.drop_duplicates()` on both columns jointly.
  De-duplicating column by column would break the x↔y pairing and under-count sites as soon as
  two sites share an x or a y — which any regular layout does. (This dataset happens to have 32
  distinct x and 32 distinct y, so the naive count coincides here; the guard is for the
  datasets where it does not.)
- **It is not cheap.** `_update_giulia_object_static()` re-processes the whole `__init__` chain,
  including `pairing_cell_obj`, which derives the cell-adjacency matrix for the new deployment.
  Expect the Dedalo constructor to dominate setup time — pass `skip_pairing=True` only if you
  know the pairing is unaffected.

### Check the dataset after loading it

A dataset is engineering parameters someone else produced. Run the checklist on it — this is
exactly what it is for.

On the Valencia dataset, straight after the Dedalo constructor:

```
[   3s] synthetic build : df_ep (114, 96), sites 19
[   6s] single tier     : df_ep (114, 96), sites 19
[ 197s] after Dedalo    : df_ep (153, 96), sites 32, UEsite cells 96
            RAT {'LTE': 93, 'NR': 60}  roles {'coverage': 93, 'capacity': 60}

  check_df_ep() on the RAW dataset: 6/11
    FAILING: _check_dl_carrier_wavelength_m, _check_dl_PRBs_available,
             _check_ul_available_PRBs, _check_ul_ofdm_symbol_duration_us,
             _check_ul_control_channel_overhead
```

The geometry is right — 153 cells, 32 sites, 96 UE-site cells, roles assigned. The
**engineering parameters are not**. Three distinct defects:

| Defect | Rows | What is wrong |
|---|---:|---|
| `dl_carrier_wavelength_m = 0.15` at 1.8 GHz | 93 (all LTE) | should be `0.1667`; 0.15 is the value for 2.0 GHz |
| `dl_bandwidth_MHz = 140` at 30 kHz SCS | 57 | not a valid NR carrier at all — the widest at 30 kHz is 100 MHz |
| `ul_subcarrier_spacing_kHz = 150` | 60 (all NR) | not an NR numerology (15/30/60/120/240/…), and the UL symbol duration and overhead still hold their 15 kHz values |

The wavelength one is not cosmetic: `dl_carrier_wavelength_m` feeds the wave vector
`(2π/λ)` in [`array_steering_vectors.py`](../../giulia/antenna/array_steering_vectors.py), so a
+11.1% wavelength error is a −10.0% error in the steering phases of every LTE cell. *(How much
that moves a final KPI is not measured here — only that the input is wrong.)*

### Repairing it — sources first, then derived

`repair_df_ep()` recomputes *derived* columns from their sources. It cannot invent a valid
source, and on this dataset it **raises**:

```
ValueError: Invalid combination (bandwidth_MHz=140.0, subcarrier_spacing_Hz=30.0).
```

So the order is: fix the invalid source values yourself, then let the updaters do the rest.

```python
df = g.network_deployment_obj.df_ep
# 1. invalid SOURCES — no updater can derive these for you
df.loc[df['dl_bandwidth_MHz'] == 140, 'dl_bandwidth_MHz'] = 100
df.loc[df['ul_subcarrier_spacing_kHz'] == 150, 'ul_subcarrier_spacing_kHz'] = 30
# 2. now every DERIVED column follows
dedalo.repair_df_ep()
dedalo.assert_df_ep_consistent()
```

Measured:

| stage | checklist |
|---|---|
| raw dataset | 6/11 |
| after fixing the two invalid sources | 8/11 |
| after `repair_df_ep()` | **11/11** |

> The reference drivers call `set_BW_scs_prb_numerology(BW_MHz_5G=100, scs_kHz_5G=60)`, which
> overwrites the 140 MHz rows as a side effect — so the downlink defect does not reach those
> campaigns. The **LTE wavelength and the uplink columns are not touched by it** and remain
> wrong unless repaired. Check, don't assume.

### Ordering constraints

These are not stylistic. Each one silently produces a different network if moved:

| Order | Why it cannot move |
|---|---|
| tier override **before** Dedalo | the dataset's UE playground is sized against the layout Giulia currently holds |
| numerology **before** appending any extra cell | `set_BW_scs_prb_numerology` selects by `RAT`; an appended NR cell would be re-numerologised along with the terrestrial ones |
| per-radio-unit randomisation **last** | both the numerology step and any deployment change end in a `power_consumption_obj.process()` sweep, which rebuilds per-RU parameters from the type tables and erases earlier noise |
| never re-process `network_deployment_obj` afterwards | it would rebuild `df_ep` from the scenario template and discard the dataset entirely |

## 6. How the real agents wire in

The agents delivered with this repository follow the loop of the previous sections —
`process_static` → `configure` → `process_realTime` → `run_simulation` — and read the results
back through the `Saveable` keys:

- [`examples/ott_integration/rl_integration.ipynb`](../../examples/ott_integration/rl_integration.ipynb):
  a Gymnasium-style reinforcement-learning environment built on a Dedalo realization.
- [`examples/ott_integration/bo_integration.ipynb`](../../examples/ott_integration/bo_integration.ipynb):
  a Bayesian-optimization evaluator that applies a candidate configuration after `process_static`
  and before `configure`, and scores it over whole episodes.

---

## 7. Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: ... should be a UE_Deployment object, not a list` | UE-side API used before `configure()` | Move the call to `process_realTime`, or after `g.configure()` |
| `TypeError: Can't instantiate abstract class` | Subclass missing `process_static` / `process_realTime` | Implement both, even as no-ops |
| `KeyError: Attribute '…' not found` | Column typo | Check against `df_ep.columns` — this error is the accessor doing its job |
| `ValueError: Length mismatch: N cells … M values` | Value list not 1 and not `len(cells)` | Broadcast only works one-value-to-many-cells |
| `RuntimeError: size of tensor a (65) must match tensor b (135)` | Derived field repaired (the default) but the caching object not re-processed | Re-`process()` that object — see [§3](#3-a-write-that-needs-repair-and-re-process) |
| Results implausibly good, no error | Repair was suppressed with `repair=False`, or `df_ep` was written directly | Drop `repair=False`; write through the setter; `assert_df_ep_consistent()` — [§4](#4-guarding-with-the-checklist) |
| `WARN ... determines 'SSB_number_of_beams', for which no updater exists` | Wrote a precoder / beam-count column, which has no repair | Recompute the dependant yourself, then `assert_df_ep_consistent()` |
| Write appears ignored | Written after `g.configure()` when the field is consumed by it | Move it into `process_static` |
| Edits vanish | `network_deployment_obj.process()` called after writing | Never re-process that object — it rebuilds `df_ep` from the scenario template |

---

## See also

- [The API contract](./api.md) — the full method reference and lifecycle rules.
- [`df_ep` consistency](./df-ep-consistency.md) — derived fields and the checklist.
- [KPIs and observations](./kpis.md) — what to read back and how it is currently reduced.
