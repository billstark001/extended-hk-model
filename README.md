# Extended HK model

Microscopic code and reproduction workflows for [Segregation Before Polarization](https://arxiv.org/abs/2601.16457).

Install the Python project and the separate runtime bindings as described in [runtime installation and data locations](docs/runtime-dependencies.md). The microscopic solver stays in [social-media-models](https://github.com/billstark001/social-media-models); the pair-measure solver stays in [social-media-mesoscopic-models](https://github.com/billstark001/social-media-mesoscopic-models). Set executable, source and data paths through the ignored `.env` or environment.

Run `python -m experiments.paper.run --help` for paper simulation groups. Analysis and plots remain in `experiments/paper`. Use `python -m experiments.paper.mechanism_cases.social_force_probe.run --help` for the microscopic mechanism probe.

Shared utilities live in `src/common`; microscopic model tools remain in `src/ehk`. The theory paper (unpublished yet) uses `src/theory` for mathematical operators and observable definitions, and `experiments/paper_theory` for parameter designs, analysis and figures. See the [experiment guide](experiments/paper_theory/README.md) and [architecture](docs/architecture.md). The [paper-code migration guide](experiments/paper_theory/MIGRATION.md) covers the migrated generation, fitting, analysis and plotting workflows. Large data stays external; generated outputs remain ignored. CAS 2026 is a literature reference for the theory workflow and supplies no required raw dataset. Prior abandoned theory studies remain archival. Use `python -m experiments.paper_theory --help` for the unified theory interface; its custom micro/pair experiments use one Go adapter.

Standalone workflow commands live in [`tools/`](tools/README.md): `hdd-bridge` archives completed SSD simulations to HDD, and `waitpid` attaches a follow-up command to an existing process.

## Development

Install `ruff` to format and check the Python packages:

```sh
python -m pip install ruff
ruff format src/common src/ehk src/theory experiments/paper_theory
ruff check src/common src/ehk src/theory experiments/paper_theory
```

The root `ruff.toml` uses 120 columns and keeps short containers compact despite trailing commas. Public API docstrings describe units, array shapes, cache identity and output side effects; experiment protocols and generated numerical results retain their existing meaning.

## Citation

If you use this code in your research, please cite:

``` bibtex
@misc{zhao2026segregation,
    title={Segregation Before Polarization: How Recommendation Strategies Shape Echo Chamber Pathways}, 
    author={Junning Zhao and Kazutoshi Sasahara and Yu Chen},
    year={2026},
    eprint={2601.16457},
    archivePrefix={arXiv},
    primaryClass={cs.SI},
    url={https://arxiv.org/abs/2601.16457}, 
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

This research investigates the critical role of algorithmic recommendations in fostering polarization on social media platforms. The findings provide theoretical frameworks for stage-dependent interventions, suggesting platforms could dynamically adjust algorithms to mitigate polarization without resorting to censorship.
