# Dedalo — The API contract

Reference for the [`Dedalo`](../../dedalo/dedalo.py#L50) base class: what it gives you, what
you must implement, when each call is legal, and the rule that decides whether a write needs
follow-up work.

Everything on this page was verified by running it against this tree
(`GiuliaStd` / `3GPPTR38_901_5G`, 57 cells, 570 UEs).

---

## 1. The class

```python
from dedalo.dedalo import Dedalo   # dedalo/ is a namespace package; import from the repo root

class Dedalo(ABC):
    def __init__(self, giulia: GiuliaLike) -> None: ...
```

`Dedalo` is an **abstract base class**. It stores one thing — a reference to the live Giulia
instance, as `self.giulia_obj` — and exposes accessors onto it. It holds no copy of the
simulator's state, so anything it writes is immediately visible to Giulia, and anything Giulia
changes is immediately visible through Dedalo.

`GiuliaLike` is `StandardGiulia | MultiLayerGiulia`, i.e. either preset is accepted.

To use it you must subclass it and implement two methods:

| Abstract method | Called | Purpose |
|---|---|---|
| [`process_static(episode_index=0, **kwargs)`](../../dedalo/dedalo.py#L65) | once per **episode**, **before** `g.configure()` | Apply everything that is fixed for the episode. |
| [`process_realTime(step_index=0, **kwargs)`](../../dedalo/dedalo.py#L69) | once per **step**, inside the step loop | Apply everything that changes within the episode. |

A subclass that implements neither cannot be instantiated — Python raises
`TypeError: Can't instantiate abstract class ... with abstract methods process_realTime, process_static`.
That is deliberate: **the lifecycle is not optional.**

---

## 2. The inherited surface

These members are what every Dedalo gets for free. They are the whole of the generic
interface; everything else in `dedalo/` is scenario-specific.

### Writers

#### `set_giulia_network_df_ep(name_cell_list, name_attribute, value_list, repair=True)`
[`dedalo/dedalo.py`](../../dedalo/dedalo.py#L73)

Writes one column of the **network** `df_ep`, addressed by cell *name*.

```python
# one cell, one value
d.set_giulia_network_df_ep("cell_0", "BS_tx_power_max_dBm", 40)

# many cells, one value  (broadcast)
d.set_giulia_network_df_ep(d.get_cell_names(None), "radio_enable_carrier_shutdown", True)

# many cells, many values  (positional, one-to-one)
d.set_giulia_network_df_ep(["cell_0", "cell_1"], "radio_unit_initial_status",
                           ["carrier_shutdown", "active"])
```

Behaviour worth knowing:

- A bare `str` cell name is promoted to a one-element list; a non-`list` value is promoted to
  a one-element list.
- **Broadcast is one-directional.** One value against *N* cells is repeated. *N* values
  against one cell is a `ValueError`.
- Raises `KeyError` if the column does not exist — so a typo fails loudly rather than
  silently adding a column. This is the main reason to prefer it over `df_ep[...] = ...`.
- Raises `TypeError` / `ValueError` if `network_deployment_obj` is not yet a `Network`.
- Matching is by the `name` column (`"cell_0"`, `"cell_1"`, …), **not** by DataFrame index
  and not by the `ID` column. A name that matches nothing is a silent no-op.

**`repair` (default `True`)** recomputes the derived columns this write invalidates, and only
those — it looks `name_attribute` up in `DERIVED_FIELD_UPDATERS` and runs the matching
updaters. For most columns there are none, so it is a no-op. It is cheap (pandas arithmetic
over the cell rows) and idempotent, since every updater recomputes from current sources.

It does **not** re-`process()` the Giulia objects that cached the table. When the written
column feeds one of them, the caller must do that — see [§5](#5-the-re-process-rule).

```python
d.set_giulia_network_df_ep(cells, "dl_bandwidth_MHz", 50)                # repaired; re-process still yours
d.set_giulia_network_df_ep(cells, "dl_bandwidth_MHz", 50, repair=False)  # legacy raw write
```

> ⚠️ The default is chosen so the **silent** failure is unreachable and the remaining one is
> **loud**. Skipping the repair produces a plausible number that is 2.08× wrong; skipping the
> re-process produces a shape-mismatch crash. `repair=True` removes the first and leaves the
> second, which is the failure you want. See
> [`df_ep` consistency § 6](./df-ep-consistency.md#6-what-dedalo-does-about-it).

Writing a column whose dependant has **no** updater (`SSB_precoder`, the beam counts) logs a
warning naming the column that just went stale — see `UNREPAIRABLE_DEPENDANTS`.

#### `set_giulia_ue_df_ep(name_ue_list, name_attribute, value_list)`
[`dedalo/dedalo.py`](../../dedalo/dedalo.py#L223)

The same contract for the **UE** `df_ep` (48 columns), addressed by UE name (`"UE_0"`, …).
Only legal **after** `g.configure()` — see [§4](#4-what-is-legal-when).

No `repair` argument: the updaters target the *network* schema and several of their source
columns do not exist in the UE table. The UE columns Dedalo writes (`ue_target_rate_Mbps`,
`traffic_generation_model`) have no derived dependants.

### Readers

#### `get_cell_names(attribute, value=None) -> list[str]`
[`dedalo/dedalo.py`](../../dedalo/dedalo.py#L277)

```python
d.get_cell_names(None)              # every cell name
d.get_cell_names("RAT", "NR")       # only 5G cells
d.get_cell_names("cell_role", "coverage")
```

Passing `attribute=None` returns all names; otherwise it returns the names where
`df_ep[attribute] == value`. This is the idiomatic way to build the `name_cell_list` for a
write — select, then set:

```python
five_g = d.get_cell_names("RAT", "NR")
d.set_giulia_network_df_ep(five_g, "dl_bandwidth_MHz", 100)
```

#### `get_ue_names(attribute, value=None) -> list[str]`
[`dedalo/dedalo.py`](../../dedalo/dedalo.py#L289) — the UE equivalent.

#### `giulia_number_of_cells` / `giulia_number_of_ue` (properties)
[`giulia_number_of_cells`](../../dedalo/dedalo.py#L302) / [`giulia_number_of_ue`](../../dedalo/dedalo.py#L317)

Row counts of the two tables. `giulia_number_of_ue` is the correct length for a per-UE vector
such as a target-rate list.

### Consistency

Thin wrappers over the check/update functions in [`tests/`](../../tests/), bound to this
instance's table — Dedalo imports them rather than reimplementing them. Full treatment in
[`df_ep` consistency](./df-ep-consistency.md).

| Method | Does |
|---|---|
| [`check_df_ep(verbose=False)`](../../dedalo/dedalo.py#L182) | Run all 11 derived-field checks; returns `{check_name: (passed, message)}`. `verbose=True` prints the coloured report. |
| [`assert_df_ep_consistent()`](../../dedalo/dedalo.py#L198) | Raise `ValueError` naming the stale columns. Cheap enough for the top of `process_static`. |
| [`repair_df_ep(fields=None)`](../../dedalo/dedalo.py#L207) | Recompute every derived column, or a named subset. |

```python
d.assert_df_ep_consistent()                 # fails fast if a derived column drifted
d.repair_df_ep(["update_dl_PRBs_from_bandwidth_and_scs"])
```

---

## 3. The lifecycle

Every caller in this repository — the RL and BO notebooks of `examples/ott_integration/` — uses
this sequence:

```mermaid
sequenceDiagram
    participant A as Agent
    participant D as Dedalo
    participant G as Giulia

    A->>G: Giulia(preset, scenario, …)
    Note over G: __init__ builds df_ep,<br/>antennas, power model,<br/>shadowing (cell side only)
    A->>D: Dedalo(g, …)
    Note over D: binds to g; may write df_ep

    loop each episode
        A->>D: process_static(episode_index, **knobs)
        Note over D: UE counts, radio status,<br/>thresholds, traffic profile
        A->>G: g.configure(ep, max_time_us)
        Note over G: builds UEs, geometry,<br/>channel, BSs, event engine

        loop each step
            A->>D: process_realTime(step_index, **knobs)
            Note over D: per-step traffic,<br/>per-UE targets
            A->>G: g.run_simulation(step, interval_us)
        end

        G-->>A: Saveable.get(...) / performance_obj.r_e_*
    end
```

The ordering constraints are hard, and each has a reason:

| Constraint | Why |
|---|---|
| `Dedalo(...)` after `Giulia(...)` | Dedalo binds to a live instance; `df_ep` must already exist. |
| `process_static` **before** `configure` | `configure` reads the UE counts and network state Dedalo just wrote, and builds the UEs from them. Writing after means the episode ignores you. |
| `process_realTime` **before** `run_simulation` | The step's settings must be in place before the event engine advances. |
| Read results **after** `run_simulation` | The step has to have run. Read through `Saveable.get`; the legacy `performance_obj` likewise accumulates per step and is empty before the first one. |

### What belongs in which method

| Put it in `process_static` | Put it in `process_realTime` |
|---|---|
| UE population size for the episode | Per-UE target rates (UEs exist only now) |
| Radio on/off status carried across episodes | Per-step traffic realization |
| Carrier-shutdown / wake-up thresholds | Per-step scheduling overrides |
| Event rescheduling cadence | Anything reacting to the previous step's KPIs |
| Anything `configure()` consumes | Anything `configure()` would overwrite |

---

## 4. What is legal when

Giulia pre-declares its component slots as empty lists
([`g_standard_preset.py`](../../giulia/presets/g_standard_preset.py#L144)) and fills them
as the pipeline advances. Dedalo's type guards convert "not built yet" into a clear
`TypeError`.

Measured on a live instance:

```
=== STATE RIGHT AFTER Giulia.__init__ (before configure) ===
  network_deployment_obj : Network
  ue_deployment_obj      : list
  base_stations_obj      : list

-- Dedalo NETWORK-side API before configure --
  get_cell_names(None)     -> OK, 57 cells, first=['cell_0', 'cell_1', 'cell_2']
  giulia_number_of_cells   -> 57
  set_giulia_network_df_ep -> OK, cell_0 BS_tx_power_max_dBm now 40

-- Dedalo UE-side API before configure --
  get_ue_names        RAISES TypeError: The ue_deployment_obj should be a UE_Deployment object, not a list.
  giulia_number_of_ue RAISES TypeError: The ue_deployment_obj should be a UE_Deployment object, not a list.
  set_giulia_ue_df_ep RAISES TypeError: ... Giulia has to be initialized first.

=== after g.configure(0, 1_000_000) ===
  ue_deployment_obj      : UE_Deployment
  base_stations_obj      : BaseStations
  get_ue_names(None)     -> OK, 570 UEs, first=['UE_0', 'UE_1', 'UE_2']
  giulia_number_of_ue    -> 570
```

**Rule of thumb: the network side is available from construction; the UE side needs
`configure()` first.**

`base_stations_obj` follows the UE side. This is why
[`set_dedalo_radio_status_from_giulia_runTime`](../../dedalo/dedalo_EE.py#L288) branches on
`isinstance(self.giulia_obj.base_stations_obj, list)`: on the very first episode it reads the
radio status out of `df_ep`, and on later episodes out of the live `BaseStations` objects.
A Dedalo that needs to survive episode 0 must handle both.

---

## 5. The re-process rule

This is the part most easily got wrong, and the reason
[Overview §5](./overview.md#5-why-a-bare-write-is-not-enough) shows a 2× error.

Giulia builds its components in two waves. **`configure()` rebuilds one wave and not the
other**, and that determines whether your write is picked up for free.

| Built in | Rebuilt by `configure()`? | Objects |
|---|---|---|
| `Giulia.__init__` | ❌ **No** | `site_deployment_obj`, `ue_playground_deployment_obj`, `ue_hotspot_deployment_obj`, `network_deployment_obj`, `pairing_cell_obj`, `cell_antenna_array_structure_obj`, `cell_SSB_conf_obj`, `cell_CSI_RS_conf_obj`, `power_consumption_obj`, `cell_re_selection_conf_obj`, `time_frequency_resource_obj`, `mcs_resource_obj`, `bler_vs_sinrs_lut_obj`, `mi_obj`, `shadowing_map_cell_obj` |
| `Giulia.configure` | ✅ **Yes**, every episode | `ue_deployment_obj`, `ue_antenna_array_structure_obj`, all `distance_angles_*`, the whole channel chain, `*_RSRP_*`, `*_sinr_*`, `ue_throughput_*`, `base_stations_obj`, `eventDriven_obj`, `performance_obj` |

### The rule

> **If the field you wrote is consumed only by an object that `configure()` rebuilds, the
> write is picked up for free. If it is consumed by an `__init__`-built object, you must
> re-`process()` that object yourself.**

Two verified examples of each side:

**Free — no re-process needed.** `radio_unit_initial_status` is consumed by
`base_stations_obj`, which `configure()` rebuilds:

```
reprocess       -> power [24.4281] kW ; shutdown cells seen = 19
no reprocess    -> power [24.4281] kW ; shutdown cells seen = 19
```

Identical. Calling `power_consumption_obj.process()` here is harmless but unnecessary.

**Not free — re-process required.** `dl_bandwidth_MHz` is consumed by
`time_frequency_resource_obj`, built in `__init__` and never rebuilt. Skipping the
re-process leaves the old PRB grid in place, and the run reports a throughput 2.08× too
high. Full truth table in [Overview §5](./overview.md#5-why-a-bare-write-is-not-enough).

### The re-process chain

When you do need it, re-process **in construction order**, because each object reads the ones
before it. `Dedalo_UnRL._update_giulia_object_static` already encodes exactly this list:

```python
for object_name in [
    'ue_hotspot_deployment_obj',
    'pairing_cell_obj',                  # expensive — skip with skip_pairing=True if unaffected
    'cell_antenna_array_structure_obj',
    'cell_SSB_conf_obj',
    'cell_CSI_RS_conf_obj',
    'power_consumption_obj',
    'cell_re_selection_conf_obj',
    'time_frequency_resource_obj',
    'mcs_resource_obj',
    'bler_vs_sinrs_lut_obj',
    'mi_obj',                            # only when sinr_mapping == "MIESM"
    'shadowing_map_cell_obj',
]:
    getattr(self.giulia_obj, object_name).process()
```

Note what is **absent**: `network_deployment_obj`. Re-processing it would rebuild `df_ep`
from the scenario template and **discard every edit you just made**. Never re-process it
after a write.

In practice you rarely need the whole chain — re-process the smallest suffix that covers the
objects reading your field. `Dedalo_UnRL._update_giulia_object_static` walks the whole list
and takes `skip_pairing=True` to omit the expensive one. The field-to-consumer mapping is in
[`df_ep` consistency § 4](./df-ep-consistency.md#4-which-object-consumes-what).

---

## 6. `Dedalo_EE` and `Dedalo_UnRL`

Two concrete subclasses ship with the repository. Neither defines the interface; both are
realizations of it.

| Class | Status | Notes |
|---|---|---|
| [`Dedalo_EE`](../../dedalo/dedalo_EE.py#L47) | **Legacy — no callers.** Not constructed anywhere in this repository. | Energy-efficiency configurator: 24-hour UE profile, HAPS/terrestrial hierarchical pairing, carrier-shutdown thresholds. Its `__init__` runs a *whole extra Giulia simulation* ([`_compute_pairing_coverage_cell_4G_only`](../../dedalo/dedalo_EE.py#L777)) to derive pairings from RSRP Wasserstein similarity. Read it for the technique, not as a template. |
| `Dedalo_UnRL` ([`dedalo/dedalo_UnRL.py`](../../dedalo/dedalo_UnRL.py)) | **Live.** | Adds data-driven deployment loading, RU parameter randomization, UE contour fields, and numerology control. Used as an example in [Worked examples § 5](./examples.md#5-example-swapping-in-a-real-network-deployment). |

Useful patterns to borrow from `Dedalo_UnRL`, regardless of what you are building:

- **`set_BW_scs_prb_numerology`** — the canonical write→repair→re-process sequence, in one
  method. It writes `dl_bandwidth_MHz` and `dl_subcarrier_spacing_kHz`, calls
  `update_dl_PRBs_from_bandwidth_and_scs` to repair the derived PRB count, then calls
  `_update_giulia_object_static(skip_pairing=True)`. This is the shape every non-trivial
  Dedalo setter should have.
- **`_update_giulia_object_static`** — the re-process chain above.
- **`process_static` defaults** — it defaults the shutdown/wake-up thresholds to `0.30 / 0.70`
  and the traffic profile to `{"mu_lin": 5.0, "sigma_lin": 4.5}` rather than `None`, so a
  caller that passes nothing still gets a well-defined episode.

---

## 7. Writing your own Dedalo — checklist

1. Subclass `Dedalo`; implement `process_static` and `process_realTime`.
2. In `__init__`, call `super().__init__(giulia)` first.
3. Write through `set_giulia_network_df_ep` / `set_giulia_ue_df_ep`, not through raw
   `df_ep` assignment — you get the column check and the broadcast rules.
4. Keep UE-side writes out of `process_static`; the UEs do not exist yet.
5. Leave `repair=True` alone. Pass `repair=False` only when you deliberately want the raw
   historical write and have a reason.
6. After writing a field consumed by an `__init__`-built object, re-`process()` that object
   yourself. If you forget, Giulia crashes with a shape mismatch rather than lying to you —
   but it still costs you the run.
7. Never re-process `network_deployment_obj`: it rebuilds `df_ep` from the scenario template
   and discards your edits.
8. Call `assert_df_ep_consistent()` at least once during development — it catches exactly the
   class of error that otherwise passes silently.
9. Handle both branches of `base_stations_obj` (`list` on episode 0, `BaseStations` after) if
   you read runtime state.

A complete worked implementation is in [Worked examples § 2](./examples.md#2-a-minimal-dedalo-from-scratch).
