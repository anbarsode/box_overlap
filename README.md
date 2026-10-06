# Probability overlap calculator

Approximate two independent sampled probability blocks with disjoint boxes, find
geometrically overlapping event pairs, and integrate their box densities in C++.

- [Running instructions and useful settings](RUNNING.md)
- [Algorithm, public parameters and binary format](bo/README.md)
- [All demonstrations and comparisons](validation/playground.ipynb)
- [Single scientific report: experiments, costs, validation and scope](validation/report.md)

Use the `gw` environment. Code lives in `bo`; tests and the notebook live in
`validation`. Raw samples are processed one event at a time. Saved compact boxes
can be reused for later overlap calculations.
