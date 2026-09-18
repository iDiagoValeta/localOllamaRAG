# Tests

| Location | Protects | Gate |
|---|---|---|
| `tests/unit/` | Domain, ports, config, application plus adapters with doubles | fast `architecture` job |
| `tests/characterization/` | Pins current pipeline behavior including known bugs, do not edit | fast `engine` job |
| `tests/eval/` | Gold cases plus grader plus baseline floor, do not weaken | manual full-eval |
| `tests/test_*.py` loose | Engine and web wiring that needs real deps | fast `engine` job |
| `harness/tests/` | Config search with fake evaluators | fast `architecture` job |

New behavior needs a test that exercises it. Fast check without GPU: `pytest tests/unit tests/eval harness/tests --ignore=tests/unit/adapters`.
