# Extended HK model

Microscopic code and reproduction workflows for [Segregation Before Polarization](https://arxiv.org/abs/2601.16457).

Install the Python project with `python -m pip install -e .`, and install the external microscopic runtime and its `smp_bindings` from [social-media-models](https://github.com/billstark001/social-media-models). Set `SMP_BINARY_PATH` and the workspace paths through `.env` or the environment.

Run `python -m experiments.paper.run --help` for paper simulation groups. Analysis and plots remain in `experiments/paper`. Use `python -m experiments.paper.mechanism_cases.social_force_probe.run --help` for the microscopic mechanism probe.

The next paper will use `src/theory` and `experiments/paper_theory`; see [architecture](docs/architecture.md). Prior abandoned theory studies and local data were archived outside this checkout.

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
