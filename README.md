# BALM experiments

Simulation code and experiment outputs for budget-aware just-in-time warm-up of serverless containers. This repository contains the reproducibility material, not the manuscript or submission documents.

## Data

Experiments use the public Azure Functions 2019 trace (CC BY 4.0): M. Shahrad et al., *Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider*, USENIX ATC 2020. The raw trace and derived sparse matrix are not committed.

```sh
python3 -m pip install -r requirements.txt
sh data/download_trace.sh /tmp/aztrace
AZ_TRACE_DIR=/tmp/aztrace/data python3 code/00_build_dataset.py
```

The dataset builder writes `data/derived/trace.npz`. By default, experiment scripts read that file and write CSV or JSON outputs under `results/`. Full-trace runs can take substantial time and memory.

## Experiments

```sh
python3 code/01_smoke_test.py
python3 code/02_predictability.py 14
python3 code/03_characterize.py
python3 code/run_main.py
python3 code/run_sensitivity.py ablation
python3 code/run_sensitivity.py lowfreq
python3 code/run_sensitivity.py coldpen
python3 code/run_learned_variants.py
python3 code/run_capacity.py --train-days 5 --tag causal-full
```

`code/sim.py` implements minute-level replay and baseline policies. `code/policies_plan.py` contains budgeted LRU, popularity, BALM, and oracle planners. `code/ml_planner.py` and `code/run_learned_variants.py` implement the trained baseline. `code/capacity_sim.py` and `code/run_capacity.py` implement the shared hard total-memory cap. `code/04_make_figures.py` and `code/make_capacity_figure.py` turn result files into plots under `figures/`.

The final hard-cap output is `results/capacity_causal-full.csv`; the final learned comparison is `results/learned_variants.csv`. Other retained CSV and JSON files document the experiment sweeps. The simulator models cold-start latency; the invocation counts, execution durations, and memory footprints come from the public trace.

## License

See [LICENSE](LICENSE). The Azure trace has its own CC BY 4.0 license.
