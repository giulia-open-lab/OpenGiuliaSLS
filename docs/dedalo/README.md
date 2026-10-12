# Dedalo

**Dedalo is the configuration interface to Giulia.** It is the layer through which an
external caller — a script, a Bayesian-optimization loop, an RL agent — modifies a live
Giulia simulation, without reaching into the simulator's internals.

Dedalo is the **write** path. Results are read back through `Saveable`
([output-saveable](../output-saveable/README.md)), not through Dedalo — see
[KPIs and observations § 1](./kpis.md#1-the-read-path).

> The name comes from **Daedalus**, the legendary builder of Greek mythology. Like its
> namesake, this component sets the values and options that let Giulia build and run
> custom simulations ([`dedalo/dedalo.py`](../../dedalo/dedalo.py)).

Start with the **Overview**, then read the topic pages in order.

| Page | What it covers |
|---|---|
| **[Overview](./overview.md)** | What Dedalo is and why it exists, where it sits in the Giulia pipeline, the "Agent Access Point" idea, and the architecture diagrams. Read this first. |
| **[The API contract](./api.md)** | The `Dedalo` base class: every method, the two-phase lifecycle, what works before vs. after `configure()`, and the rule for when a write needs a re-process. |
| **[`df_ep` consistency](./df-ep-consistency.md)** | The engineering-parameter table Dedalo writes into, its derived fields, the checklist/updater machinery, and a measured demonstration of what a naive write costs. |
| **[KPIs and observations](./kpis.md)** | How a Dedalo caller reads results back — through `Saveable`, not through Dedalo — what the legacy `r_e_*` surface is, and why the reward/observation code duplicated across agents needs one definition. |
| **[Worked examples](./examples.md)** | Runnable, verified examples: a minimal Dedalo from scratch, the canonical episode/step loop, and how the BO and RL agents wire in. |

## Fast map of the code

| Concern | File | Key symbol(s) |
|---|---|---|
| The interface itself | [`dedalo/dedalo.py`](../../dedalo/dedalo.py) | [`Dedalo`](../../dedalo/dedalo.py#L50) (abstract base) |
| A legacy realization (no callers) | [`dedalo/dedalo_EE.py`](../../dedalo/dedalo_EE.py) | [`Dedalo_EE`](../../dedalo/dedalo_EE.py#L47) |
| A concrete realization (example only) | [`dedalo/dedalo_UnRL.py`](../../dedalo/dedalo_UnRL.py) | `Dedalo_UnRL` |
| The table Dedalo writes | [`giulia/bs/bs_deployments.py`](../../giulia/bs/bs_deployments.py) | `Network.df_ep` |
| Derived-field checks | [`tests/network_df_ep_check_list.py`](../../tests/network_df_ep_check_list.py) | [`run_all_checklist`](../../tests/network_df_ep_check_list.py#L458) |
| Derived-field dependency map | [`tests/network_df_ep_update.py`](../../tests/network_df_ep_update.py) | [`DERIVED_FIELD_UPDATERS`](../../tests/network_df_ep_update.py#L171), [`repair_dependants_of`](../../tests/network_df_ep_update.py#L213) |
| Derived-field repair | [`tests/network_df_ep_update.py`](../../tests/network_df_ep_update.py) | [`run_all_updates`](../../tests/network_df_ep_update.py#L225), [`run_selected_updates`](../../tests/network_df_ep_update.py#L233) |
| What Giulia rebuilds per episode | [`giulia/presets/g_standard_preset.py`](../../giulia/presets/g_standard_preset.py) | [`Giulia.configure`](../../giulia/presets/g_standard_preset.py#L377), [`Giulia.run_simulation`](../../giulia/presets/g_standard_preset.py#L1261) |
| Results read back | [`giulia/outputs/saveable.py`](../../giulia/outputs/saveable.py) | `Saveable.get`, `Saveable.catalog` — the read path |
| Legacy fixed KPI surface | [`giulia/outputs/save_performance.py`](../../giulia/outputs/save_performance.py) | the `r_e_*` attributes on `performance_obj` |

## The contract in six lines

Steps 1–4 are Dedalo's whole runtime contract, and the agents delivered with this repository use
exactly that sequence. Step 5 is the read path, and belongs to `Saveable`:

```python
dedalo.process_static(episode_index=ep, **episode_knobs)   # 1. set up the episode
g.configure(ep, max_simulation_time_us)                    # 2. Giulia builds UEs + links
for step in range(number_of_steps):
    dedalo.process_realTime(step, **step_knobs)            # 3. set up the step
    g.run_simulation(step, decision_making_interval_us)    # 4. advance the simulation

# 5. read back — through Saveable, by declared key
p_ru = Saveable.get("Power_Consumption.power_consumption_per_radio_unit_incl_bbu_kW")
```

> The agents in this tree still read step 5 from the legacy `g.performance_obj.r_e_*`
> attributes. Both work; only the `Saveable` keys pick up newly declared quantities without an
> exporter change. See [KPIs and observations § 1](./kpis.md#1-the-read-path).

See [The API contract § Lifecycle](./api.md#3-the-lifecycle) for what each call is allowed
to touch, and [Worked examples](./examples.md) for complete runnable programs.

## Status of the code, as of this writing

These are facts about the current tree, not recommendations. They matter because they
determine which parts of the documentation describe *what is* and which describe *what is
intended*.

- [`Dedalo`](../../dedalo/dedalo.py#L50) is an **abstract base class** with two abstract
  methods, `process_static` and `process_realTime`. It cannot be instantiated directly.
- [`Dedalo_EE`](../../dedalo/dedalo_EE.py#L47) **is never constructed anywhere** in either
  repository. It is superseded by `Dedalo_UnRL` and should be read as legacy.
- `Dedalo_UnRL` is the only concrete realization with live callers. It is **one
  realization**, not the definition of Dedalo — this documentation treats it as an example.
- The derived-field checks and repairs live under [`tests/`](../../tests/). `Dedalo` does not
  reimplement them — it imports them and exposes them as methods bound to its own instance
  (`check_df_ep`, `assert_df_ep_consistent`, `repair_df_ep`), and `set_giulia_network_df_ep`
  repairs derived columns by default. See
  [`df_ep` consistency § 6](./df-ep-consistency.md#6-what-dedalo-does-about-it).
- The `dedalo/` package has **no `__init__.py`**; it is imported as a namespace package
  from the repository root.
