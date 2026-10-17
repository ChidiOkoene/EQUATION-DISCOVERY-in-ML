# Datasets

| Folder | Description |
|--------|-------------|
| `benchmarks/burgers/` | Burgers PDE `.mat` / CSV inputs for the benchmark experiments |
| `applications/industrial/` | Clamping-force and SWE_VDC / ABSTCSVDC time-series CSVs for industrial PiDL |
| `tabular/` | Public tabular samples used in early trials (housing, wine, automobile) |
| `reconstructed/` | Feature rows synthesized to match discovered formulas ([`tools/data_reconstruct_mainX.py`](../tools/data_reconstruct_mainX.py)) |

Point experiment code at these paths with relative imports from `experiments/{problem}/{method}/`.
