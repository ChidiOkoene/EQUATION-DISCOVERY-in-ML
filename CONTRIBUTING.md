# Contributing

This repo is primarily a research showcase. When adding work:

1. Put **experiments** under `experiments/{problem}/{method}/` (problem-first: `burgers` or `industrial`, then `pidl` / `pysr` / `sindy` / `pinns`).
2. Put **datasets** under `data/` (`benchmarks/burgers`, `applications/industrial`, `tabular`, or `reconstructed`).
3. Put **run outputs** under `results/` mirroring the experiment path — do not dump plots at the repo root.
4. Put didactic tutorials under `examples/`; papers and conceptual notes under `docs/`.
5. Prefer **relative paths** into `data/` so experiments run on any machine.
6. Update the README gallery or “Explore by path” section if you add a headline result.
