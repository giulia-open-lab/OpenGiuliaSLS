# Saveable Concurrency — the per-context registry

This page documents **where the Saveable registry lives** and what that means when more
than one thing runs at once: several simulations in parallel, or a background thread
reading KPIs while a simulation runs.

It is the reference to read when a Saveable "disappears", when `Saveable.get()` returns
`None` from a thread you know registered the object, or before you start a simulation on
a thread of your own.

If you are here to understand **why the design looks like this** — including the mistake an
earlier version of it made and what that cost — read
[§11 Design history](#11-design-history-three-attempts-at-one-registry). It is the record of
the decisions, kept so that they are not silently undone.

Companion pages:

- [Saveable Objects](../source/saveable-objects.rst) — how a class *becomes* a Saveable.
- [Saveable Output & Export](saveable-output.md) — how registered variables become files.
- [Saveable Monitor](SaveAble_monitor.md) — the built-in live KPI reader, which depends on everything below.

---

## 1. The problem this solves

The registry was, until this change, a plain module-level list shared by the whole process:

```python
saveables: List[Saveable] = []      # the old design
```

Registration is **"latest wins"**: `Saveable.__init__` first calls
`Saveable._unregister(self.name)` to drop any earlier instance of the same class, then
appends itself. That rule is required *within* a run — the simulation re-creates its
Saveables every episode inside `configure()`, and the registry must point at the current
episode's objects, not episode 0's (see
[Saveable Objects → Multiple instances](../source/saveable-objects.rst)).

On a shared list, that same rule is destructive **across** runs. Two simulations running
in parallel both construct, say, `SINR_CRS`. Each construction evicts the *other*
simulation's instance:

```
thread A            shared list                     thread B
--------            -----------                     --------
SINR_CRS()   ->     [A:SINR_CRS]
                    [B:SINR_CRS]              <-    SINR_CRS()      (A's entry evicted)
                    [A:SINR_CRS]              <-    (next episode) A re-registers, B evicted
export      ->      reads whichever instance won the last race
```

The consequences were silent and data-dependent: `save_episode_performance` exported
columns belonging to the wrong simulation, per-entity row counts came from the wrong
network, and a run could crash outright when the surviving instance had a different
entity count than the file being written.

Locking would not fix this. The two runs do not need *serialised* access to one registry —
they need **different registries**.

---

## 2. The model: one registry per execution context

The registry now lives in a [`ContextVar`](https://docs.python.org/3/library/contextvars.html)
([`giulia/outputs/saveable.py`](../../giulia/outputs/saveable.py)):

```python
class _RegistryState:
    __slots__ = ('saveables', 'enabled')
    def __init__(self):
        self.saveables: List['Saveable'] = []
        self.enabled: bool = True

_state: ContextVar[_RegistryState] = ContextVar('giulia_saveable_state')
```

A `ContextVar` holds a **different value per execution context**, and **a thread starts with
an empty one**. Nothing is inherited from the thread that started it, so:

- every thread that runs a simulation gets its **own** registry, on first access;
- `_unregister` can only ever remove entries from the caller's own registry;
- no lock is needed, because no two runs touch the same list.

The registry and the enable flag are kept together in one `_RegistryState` object. That is
deliberate: they are two halves of the same per-run state, and §5 hands **both** to a
helper thread in one step.

> **Terminology.** "Context" here is Python's `contextvars` context, not a Giulia concept.
> In practice: *one thread = one context = one registry*, for the whole life of that thread.

Two things follow from "for the whole life of that thread", and both surprise people:

- **A context is not a run.** The state stays bound until the thread ends, so a **reused**
  worker thread — anything from a `ThreadPoolExecutor` — begins its second simulation holding
  the first one's registry. One thread per run, or none of this applies.
- **`asyncio` tasks are *not* the same case.** A task **copies** the context that created it
  rather than starting empty, so once `_state` is bound a task shares its creator's
  `_RegistryState` — see §9.

### What is per context and what is not

| Thing | Scope | Why |
|---|---|---|
| The registry (`Saveable.registry()`, `catalog()`, `get()`, `exists()`) | **per context** | so parallel simulations cannot evict each other |
| The enable flag (`Saveable.enable()` / `disable()` / `is_enabled()`) | **per context** | a nested throwaway run that disables collection must not silence a *parallel* run |
| The `Saveable` classes themselves | process-wide | they are just classes; only *instances* are registered |
| Output files, `config.yml`, the output directory | process-wide | concurrent runs still need distinct `output.directory` values (see §6) |

---

## 3. The trade-off: a fresh thread starts empty

Isolation cuts both ways. A new thread's context does **not** inherit the parent's, so a
thread that did not run the simulation sees an empty registry:

```python
Kpi()                                   # registered on the main thread

def worker():
    print(Saveable.get('Kpi.value'))    # None — different context, empty registry

threading.Thread(target=worker).start()
```

This is the single most important thing to know about the design, because the failure is
**silent**: `Saveable.get()` returns `None` for an unknown key and for an empty registry
alike, so an unwrapped reader thread just prints `None` forever.

That is exactly what §5 exists for.

---

## 4. How a simulation gets its registry

Nothing to do. The registry is created lazily on first access, so simply running a
simulation on a thread gives that run a private registry:

```python
def run_one(config):
    giulia_main, args = config.run()
    giulia_main(args, config)           # every Saveable built here registers in THIS thread

threading.Thread(target=run_one, args=(config_a,)).start()
threading.Thread(target=run_one, args=(config_b,)).start()
```

Both runs register, collect and export independently.

---

## 5. Reading a registry from another thread: `Saveable.share_registry`

To let a helper thread observe a simulation's registry, wrap its entry point:

```python
from giulia.outputs.saveable import Saveable

def poll():
    while not stop_event.is_set():
        print(Saveable.get('Cell_Selection_SSB.best_serving_cell_ID_per_ue'))
        stop_event.wait(1.0)

worker = threading.Thread(target=Saveable.share_registry(poll), daemon=True)
worker.start()
```

`share_registry` captures the `_RegistryState` of the thread that **calls** it and binds
the wrapped callable's context to that same object. Two consequences worth stating
explicitly:

- **It is live, not a snapshot.** The registry is mutated in place and never rebound, so
  Saveables registered *after* the worker started are visible to it — which is what makes
  monitoring across episodes work at all.
- **The enable flag travels with it**, since both live in the same `_RegistryState`.

Two rules:

1. **Call it on the thread that owns the registry** — the thread that runs, or is about to
   run, the simulation. `Saveable.share_registry(f)` evaluated on the *wrong* thread shares
   the wrong (probably empty) registry.
2. **Call it before starting the thread.** The capture happens when `share_registry` is
   called, not when the wrapper runs.

```
main thread (simulation)                worker thread
------------------------                -------------
state = _RegistryState()
  registry: [Kpi]
share_registry(poll)  ──── captures ───► binds the SAME state object
Kpi() (episode 1)  ── mutates in place ─► poll() sees episode 1's instance
```

### Reading from a *process*, not a thread

`share_registry` is for threads, and only because threads share one heap: the wrapper hands
over a *reference* to the same `_RegistryState` object. A separate **process** has its own
interpreter, its own `giulia.outputs.saveable` module and its own `_RegistryState`; there is
no object to hand over and no design of this registry could produce one. Live cross-process
reads therefore need an **explicit IPC mechanism** — a `multiprocessing` queue or pipe, a
socket, a file the run writes as it goes — into which the simulation process publishes what
it wants observed.

Unless you need the values *live*, do not build that: export to Parquet and read it with
[`OutputReader`](reading-parquet.md). That is what
[`scripts/generate_saveable_catalog.py`](../../scripts/generate_saveable_catalog.py) relies on
(§6), and what `SeparateProcessTests` exercises (§10).

| | Threads | Processes |
|---|---|---|
| Memory | one heap, shared | separate, nothing shared |
| Registry by default | one per thread, isolated | one per process, unreachable |
| Can they be joined up? | yes — `share_registry`, one call | only through explicit IPC, or files |
| Failure mode if you forget | reader sees an empty registry, silently | there is nothing to forget: no API pretends otherwise |

---

## 6. Running simulations in parallel

The registry no longer stands in the way, but two things still do:

- **Give each run its own `output.directory`.** Output paths are ordinary configuration,
  not per-context state; two runs pointed at one directory will overwrite each other's files.
- **Watch the C++ backends.** [`main.py`](../../main.py) runs the simulation on the OS main
  thread because PyTorch / Mitsuba must be initialised and torn down there. Moving a
  simulation onto a worker thread is safe from the Saveable side, but you inherit whatever
  constraints those backends impose.

For strong isolation — separate memory, separate GPU context, no shared globals of any
kind — use **separate processes**. That is what
[`scripts/generate_saveable_catalog.py`](../../scripts/generate_saveable_catalog.py) does:
it runs each preset in its own subprocess. Note the reason is unchanged by this design.
The catalog script runs its presets *sequentially on one thread*, so they would share that
thread's context and the first preset's Saveables would leak into the second catalog; a
subprocess per preset is still the correct fix.

---

## 7. Effect on `enable()` / `disable()`

`Saveable.disable()` stops *new* instances from registering; it never removes existing
ones. Both it and `is_enabled()` now act on the calling context.

What this changes:

- A nested throwaway run that disables collection (for example the cell-pairing sub-run,
  `cell_pairing.PairingCell._init_and_run_giulia_pairing`) can no longer silence a
  simulation running **in parallel**.
- Within one context nothing changed: the flag is still a single boolean with no stack, so
  a nested run on the *same* thread still leaks its disabled state to the caller unless the
  caller saves and restores it:

  ```python
  parent_enabled = Saveable.is_enabled()
  # ... run the nested simulation, which may call Saveable.disable() ...
  Saveable.enable() if parent_enabled else Saveable.disable()
  ```

`Simulation_Config.enable_saveable` is still read from `Saveable.is_enabled()` at
construction time, so it reflects the flag of the context that built it — do not use it to
restore another context's state.

---

## 8. Migration: what the old names became

The module-level globals were **removed**, not left as empty stand-ins. Reading them raises
an `AttributeError` naming the replacement:

```
AttributeError: 'giulia.outputs.saveable.saveables' was removed: the Saveable registry is
now per execution context, not process-global. Use Saveable.registry() instead.
```

> **Which form shows that message.** Attribute access (`saveable_module.saveables`) raises the
> `AttributeError` above, text and all. The `from giulia.outputs.saveable import saveables`
> form — which is what every old call site actually wrote — still fails immediately, but Python
> converts the module's `AttributeError` into a bare
> `ImportError: cannot import name 'saveables' from 'giulia.outputs.saveable'` and discards the
> custom text. The failure stays loud, which is the point; the instructions are on this page.

| Old | New | Notes |
|---|---|---|
| `from giulia.outputs.saveable import saveables` | `Saveable.registry()` | the live list for the current context; treat as read-only |
| `saveable_enabled` (module global) | `Saveable.is_enabled()` | |
| `for obj in saveables: ...` | `for obj in Saveable.registry(): ...` | |
| a bare `threading.Thread(target=reader)` | `threading.Thread(target=Saveable.share_registry(reader))` | otherwise the reader sees nothing (§3) |

The loud failure is the point. Three call sites read that global — the
[`Monitor`](SaveAble_monitor.md), `log_saveable_available_results` in
[`giulia/tools/tools.py`](../../giulia/tools/tools.py), and the
[catalog generator](../../scripts/generate_saveable_catalog.py) — and all three are migrated
with this change. Had the global instead been kept as an empty stand-in "for compatibility",
every one of them would have gone on importing a registry that never received anything,
reporting "0 saveable objects registered" and an empty catalog without raising a thing.
That is not hypothetical — see [§11](#11-design-history-three-attempts-at-one-registry).

---

## 9. Caveats

- **A shared registry is shared mutable state.** While a simulation re-registers a class
  (`_unregister` then `append`, two steps), a reader thread can momentarily observe that
  class as absent and get `None` for one poll. Values are read live with `getattr`, so a
  reader can also catch a half-updated set of attributes. Neither corrupts anything — for
  monitoring, poll again; for anything that must be exact, use the exported Parquet.
- **`share_registry` does not chain.** A thread started *from* a wrapped worker gets its own
  empty context; wrap that one too.
- **`asyncio` tasks are not isolated the way threads are.** A thread starts from an empty
  context, but a task starts from a **copy** of the context that created it. Copying the
  mapping copies the *reference*, so a task created after `_state` was bound shares its
  creator's `_RegistryState` — it can register into it, and evict from it, exactly as the
  creator can. Only a task created before anything touched the registry gets a state of its
  own, and then its `_state.set()` stays inside the task: what it registers is invisible to
  the code that awaited it. Neither behaviour is what "one registry per task" would suggest.
  Giulia itself runs no `asyncio`; if you add some, put each simulation on its own **thread**.
- **A reused thread keeps its registry.** The binding lasts as long as the thread, not as long
  as the run, so a second simulation submitted to the same pooled worker starts on the first
  one's registry and evicts its entries class by class. Give each run its own thread.
- **The registry is never cleared automatically.** Within one context it keeps one entry per
  class name for the life of that context, so a catalog taken after a run lists everything
  the run ever instantiated.

---

## 10. Verifying the behaviour

[`tests/test_saveable_registry.py`](../../tests/test_saveable_registry.py) pins every claim
on this page — isolation between threads, no eviction under concurrent registration,
`share_registry` visibility (including registrations made after the worker starts), the
per-context enable flag, and the `AttributeError` on the removed globals:

```bash
python -m unittest tests.test_saveable_registry -v
```

The suite runs in CI on every push, alongside the other unit tests
([`.github/workflows/unit-tests.yml`](../../.github/workflows/unit-tests.yml)):

```bash
python -m unittest tests.unit_tests tests.test_saveable_registry
```

See [Testing the Saveable registry](saveable-registry-tests.md) for what each test pins, when to
run it, and how to extend it.

That suite uses synthetic Saveables and never starts a simulation, so it cannot tell whether the
*exporters* still find anything. That is covered by
[`tests/test_saveable_pipeline.py`](../../tests/test_saveable_pipeline.py) and
[`tests/test_saveable_pipeline_concurrency.py`](../../tests/test_saveable_pipeline_concurrency.py),
which run real minimal runs through the whole chain: parallel simulations with different
deployments (including a stress case with eight at once, in threads and in processes), a live
`Monitor` on a real KPI, the catalog generator end to end, and the cell-pairing sub-run's
`disable()` / restore. See [Testing the Saveable pipeline](saveable-pipeline-tests.md):

```bash
python -m unittest tests.test_saveable_pipeline tests.test_saveable_pipeline_concurrency
```

---

## 11. Design history: three attempts at one registry

This section records **why** the registry is built this way, and in particular why it is *not*
built the simpler way that suggests itself first.

Three designs are described. Giulia went from Stage 0 straight to Stage 2 — but Stage 1 was
built and shipped in a downstream fork (OpenGiuliaSLS), where it looked correct, passed review,
and silently disabled three features. It is recorded here because it is the obvious fix, and
therefore the one most likely to be reintroduced by someone simplifying this file.

### Stage 0 — one process-global list

```python
saveables: List[Saveable] = []      # module-level, shared by everything
```

Correct for a single simulation per process, which is all Giulia originally did. It breaks as
soon as two simulations run at once, because registration is "latest wins": each construction
of a class evicts the other run's instance of that class. [§1](#1-the-problem-this-solves) has
the full walkthrough and the symptoms (columns from the wrong simulation, wrong per-entity row
counts, crashes on mismatched entity counts).

### Stage 1 — thread-local storage (the trap)

The obvious fix: swap the module-level list for `threading.local()`. **Do not do this.**

**What it gets right — and this is the hard part:**

- It correctly identified that the problem is `_unregister` operating on shared state, not
  unsynchronised access. A mutex would have been the reflexive fix and would have been *useless*:
  two runs do not need serialised access to one registry, they need two registries.
- It therefore replaced the shared list with per-thread storage, which needs no locking at all.
- The registry accessor became a function (`_get_saveables()`), so the storage could later change
  without touching the exporters. That is why Stage 2 was a small change.

**What it misses.** It reasons only from the *writer's* side — the simulation that registers
Saveables. But the registry has **readers too, and they live on other threads by design**. A
thread-local registry has no way to share, so every reader outside the simulation thread silently
sees an empty registry. In the fork that shipped this, the effect was:

| Reader | Where it runs | Symptom under Stage 1 |
|---|---|---|
| [`Monitor`](../../giulia/outputs/monitor.py) | a background daemon thread, **by design** | every KPI read `None` forever; at shutdown it warned that *all* requested KPIs "may be misspelled" |
| `log_saveable_available_results` ([`tools.py`](../../giulia/tools/tools.py)) | the simulation thread — but it read the **legacy global** | always logged "There are **0** saveable objects registered" |
| [`generate_saveable_catalog.py`](../../scripts/generate_saveable_catalog.py) | a worker subprocess — also read the **legacy global** | would regenerate [`saveable-catalog.md`](saveable-catalog.md) as an **empty** catalog |

**Why none of this raised an alarm.** Each of those failures is indistinguishable from a
legitimate state:

- `Saveable.get()` returns `None` for an unknown key *and* for an empty registry. The monitor
  printing `None` looks exactly like a typo in a KPI name — which is what its own warning then
  claimed it was.
- An empty catalog looks exactly like "the simulation has not started yet", which is a real and
  expected state ([§3](#3-the-trade-off-a-fresh-thread-starts-empty)).
- "0 saveable objects registered" looks exactly like a configuration that enables no Saveables.

**The decision that caused the silence.** Stage 1 also kept the old global in place as an empty
list "for backward compatibility":

```python
saveables: List[Saveable] = []  # kept for backward-compat imports; not used internally
```

That single line turned three would-be `ImportError`s — which any run would have raised
immediately, at import time, in the developer's face — into three wrong answers that a run
reports as normal output. **A compatibility shim that returns an empty value does not preserve
compatibility; it converts a loud failure into a silent one.**

### Stage 2 — `ContextVar` plus explicit sharing (current)

Four changes, each answering one of the above.

**1. `ContextVar` instead of `threading.local`.** Both isolate per thread, but a `ContextVar` can
be *deliberately propagated*: a helper thread can be bound to the same state object, which
`threading.local` cannot express at all. That propagation is the whole difference — it is the
escape hatch Stage 1 lacked. (It is not chosen for `asyncio`, which Giulia does not use and
which a `ContextVar` isolates only partially — [§9](#9-caveats).)

**2. `Saveable.share_registry` — sharing is opt-in and explicit.** Isolation is the default
because it is the safe answer; sharing is a single, greppable call at the one place that needs it
([§5](#5-reading-a-registry-from-another-thread-saveableshare_registry)). `Monitor` now wraps its
polling loop with it, which is what makes live monitoring work at all.

**3. The enable flag moved into the same per-context state.** `saveable_enabled` was still a
process-wide global after Stage 1 — the *same bug class*, one layer down: a nested throwaway run
calling `Saveable.disable()` (the cell-pairing sub-run does exactly this) would silence a
simulation running in parallel, producing empty KPI Parquets in a run that never touched pairing.
Registry and flag now live in one `_RegistryState`, so they are isolated together and
`share_registry` propagates both ([§7](#7-effect-on-enable--disable)).

**4. The globals were removed, not emptied.** `saveables` and `saveable_enabled` now raise an
`AttributeError` naming their replacement ([§8](#8-migration-what-the-old-names-became)), instead
of lingering as empty stand-ins. The three call sites that read them are migrated in the same
change; anything outside this repository that still imports them fails loudly, immediately, and
with instructions.

### The principle worth keeping

> **Isolation and observation are opposing requirements.** A registry that several simulations
> write to must be isolated; a registry that monitors and tooling read from must be reachable
> from another thread. A design serving only one of those silently breaks the other. Isolate by
> default, share explicitly.

And the operational lesson:

> **Narrowing the scope of a global makes every reader of that global a call site to re-audit.**
> Search for the name before changing what it means — and let the ones you miss fail loudly.

[§10](#10-verifying-the-behaviour) pins all of this in tests — including the Stage 1 trap — so a
change that reintroduces either failure mode fails the suite instead of the next experiment.
