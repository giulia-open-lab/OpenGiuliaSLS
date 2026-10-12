# Testing the Saveable registry

How to run [`tests/test_saveable_registry.py`](../../tests/test_saveable_registry.py), and —
more importantly — **why it exists at all**.

> **Looking for the tests that run an actual simulation?** Those are
> [`tests/test_saveable_pipeline*.py`](../../tests/test_saveable_pipeline.py) —
> see [Testing the Saveable pipeline](saveable-pipeline-tests.md). This page is about the
> fast, simulation-free tests of the registry itself.

It is a small, fast test file (8 tests, well under a second, no data fixtures). It is also the
only mechanism that can catch the class of bug described in
[Saveable Concurrency §11](saveable-concurrency.md#11-design-history-three-attempts-at-one-registry),
because that bug is **invisible at runtime**.

---

## 1. Why this file exists

### The failures it guards against do not look like failures

Every way the Saveable registry can break produces a result that is indistinguishable from a
legitimate state:

| What actually went wrong | What you see |
|---|---|
| Reading the registry from the wrong thread | `Saveable.get()` returns `None` — exactly like a misspelled KPI key |
| The registry was never populated in this context | an empty catalog — exactly like "the simulation has not started yet", which is a real state |
| A parallel run evicted your Saveable | a column of the wrong data, or a plausible-looking file with the wrong row count |

None of these raise. None of these log an error. A run completes, writes files, and reports
success. This is not hypothetical: in a downstream fork that adopted a thread-local registry,
exactly three features — the `Monitor`, `log_saveable_available_results`, and the catalog
generator — silently returned "nothing" for weeks, and every run looked normal
([§11](saveable-concurrency.md#11-design-history-three-attempts-at-one-registry)).

**A test suite is the only place where "empty because nothing is registered yet" can be
distinguished from "empty because you are on the wrong thread."** At runtime the two are the
same value.

### It pins two requirements that pull against each other

The registry has to satisfy two opposing constraints at once:

1. **Isolation** — parallel simulations must not see or evict each other's Saveables.
2. **Observation** — a monitor thread must be able to read a simulation's registry.

A change that serves one silently breaks the other. Testing only isolation is what let the
monitor break in the fork referenced above. So the file deliberately tests **both directions**,
including the negative case
([`test_unwrapped_worker_sees_nothing`](../../tests/test_saveable_registry.py)) — the trap is
pinned as a test so that it stays a documented, deliberate behaviour rather than a surprise.

### It is a test of decisions, not just of code

Each test corresponds to a promise made in the documentation. If someone later swaps the
`ContextVar` for something simpler, or restores the module-level `saveables` list "for
convenience", the suite fails and names the reason — instead of the next experiment producing
quietly wrong numbers.

---

## 2. What each test pins

| Test | The promise it protects | What breaks if it regresses |
|---|---|---|
| `test_registry_is_isolated_between_threads` | a new thread starts with its own empty registry | parallel runs mix each other's data |
| `test_concurrent_registration_does_not_evict` | simultaneous registration keeps both entries | the original bug returns: `_unregister` deletes another run's Saveable |
| `test_share_registry_exposes_the_creating_threads_registry` | `share_registry` shares a **live** registry, including later registrations | the monitor sees only episode 0, or nothing |
| `test_unwrapped_worker_sees_nothing` | an unwrapped thread sees nothing — *by design* | the trap becomes undocumented and gets rediscovered the hard way |
| `test_enable_flag_is_per_context` | `disable()` in one run does not silence another | a nested pairing sub-run empties a parallel run's KPI Parquets |
| `test_share_registry_also_shares_the_enable_flag` | registry and flag travel together | a shared worker disagrees with its simulation about whether collection is on |
| `test_latest_instance_wins_within_a_context` | re-creating a Saveable each episode tracks the newest instance | exported data freezes at episode 0 |
| `test_removed_module_globals_raise_a_helpful_error` | the removed globals fail **loudly** | a reintroduced empty `saveables` shim makes failures silent again |

---

## 3. How to run it

From the repository root, in the project environment (it imports `pyarrow` via
`giulia.outputs.saveable`, so the `giulia` conda environment must be active):

```bash
# the whole file
python -m unittest tests.test_saveable_registry -v

# one test, when you are iterating on a specific behaviour
python -m unittest tests.test_saveable_registry.SaveableRegistryTests.test_unwrapped_worker_sees_nothing

# by discovery
python -m unittest discover -s tests -p "test_saveable*.py"
```

Expected output:

```
Ran 8 tests in 0.005s

OK
```

No precomputed files, no shadowing maps, no GPU, no config — unlike the regression suite (§4).
That is deliberate: a test you can run in under a second is a test you will actually run before
committing.

---

## 4. Where it fits among Giulia's tests

Giulia has two separate kinds of test, with very different costs:

| Suite | What it checks | Needs | Runtime |
|---|---|---|---|
| [`tests/unit_tests.py`](../../tests/unit_tests.py) + [`tests/test_saveable_registry.py`](../../tests/test_saveable_registry.py) | small units that run no simulation — the logger, Parquet comparison helpers, and the Saveable registry under threading | [`tests/requirements.txt`](../../tests/requirements.txt) (numpy, pyarrow) | < 1 s |
| [`tests/run_tests.py`](../../tests/run_tests.py) | full-simulation regression against version-controlled golden references | a full Giulia environment | minutes |

The regression suite is what [Running Tests](../testing.md) and
[`tests/README.md`](../../tests/README.md) document. This page covers the registry tests, which
are a different tool for a different job: they do not run a simulation at all, they exercise the
registry directly.

---

## 5. Continuous integration

This suite **runs in CI on every push**, alongside the other unit tests
([`.github/workflows/unit-tests.yml`](../../.github/workflows/unit-tests.yml)):

```yaml
      - run: pip install -r tests/requirements.txt
      - run: python -m unittest tests.unit_tests tests.test_saveable_registry
```

No simulation is run, so the job stays fast. It does, however, need more than `pyarrow`:
importing `giulia.outputs.saveable` runs [`giulia/outputs/__init__.py`](../../giulia/outputs/__init__.py),
which eagerly imports `utils` and `read_outputs`, and **both import `pandas`**. That is why
[`tests/requirements.txt`](../../tests/requirements.txt) pins `pandas` alongside `numpy` and
`pyarrow` — without it the CI job fails at import with `ModuleNotFoundError: No module named
'pandas'`, before a single test runs.

Run the same command locally before pushing:

```bash
python -m unittest tests.unit_tests tests.test_saveable_registry
```

---

## 6. When to run it

Run it before committing changes to:

- [`giulia/outputs/saveable.py`](../../giulia/outputs/saveable.py) — any change to the registry,
  `share_registry`, or the enable flag;
- [`giulia/outputs/monitor.py`](../../giulia/outputs/monitor.py) — especially how the thread is
  created;
- anything that calls `Saveable.enable()` / `disable()`, such as the presets and the cell-pairing
  sub-run;
- any code that starts a thread which reads Saveables.

---

## 7. Extending it

The file provides two helpers you should reuse:

- **`_run_in_thread(target)`** — runs `target` on a fresh thread (that is, a fresh context) and
  returns its result. This is how you assert "what does another simulation see?".
- **`_Kpi` / `_OtherKpi`** — minimal Saveables. Two distinct class names, because the registry
  de-duplicates by class name; use `_OtherKpi` when a test needs two entries at once.

Note `setUp`:

```python
def setUp(self):
    Saveable.enable()
    Saveable.registry().clear()
```

The test runner's own thread is itself a context, and it is shared by every test in the file.
The registry is never cleared automatically
([Saveable Concurrency §9](saveable-concurrency.md#9-caveats)), so each test must start from a
clean one — otherwise a Saveable registered by an earlier test leaks into the next and the tests
pass or fail depending on their order. Any new test class in this file needs the same `setUp`.

When adding a test, prefer asserting on **both** sides of a behaviour — what the owning thread
sees *and* what another thread sees. A one-sided assertion is what let the original regression
through.

---

## 8. Reading a failure

| Failure | Likely cause |
|---|---|
| `a new thread must start with an empty registry` | the registry became shared again (a module global, or a `ContextVar` default that is a single mutable object) |
| `test_concurrent_registration_does_not_evict` fails | `_unregister` is reaching outside the caller's context |
| `the shared registry must stay live, not be a snapshot` | `share_registry` copied the list instead of binding the same object, or something rebinds the ContextVar instead of mutating in place |
| `test_unwrapped_worker_sees_nothing` fails | a *new* leak: the registry is visible where it should not be — check what made it reachable |
| `the enable flag must not leak across simulations` | the flag was moved back out of `_RegistryState` into a module global |
| `test_removed_module_globals_raise_a_helpful_error` fails | someone reintroduced `saveables` / `saveable_enabled` — see [§11](saveable-concurrency.md#11-design-history-three-attempts-at-one-registry) for why that is worse than it looks |

---

## See also

- [Saveable Concurrency](saveable-concurrency.md) — the design these tests pin, and its history.
- [Saveable Objects](../source/saveable-objects.rst) — how a class becomes a Saveable.
- [Running Tests](../testing.md) — the full-simulation regression suite.
