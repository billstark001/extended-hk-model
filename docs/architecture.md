# Repository architecture

`src/common` contains model-independent path settings, I/O, hashes, numerical and statistical tools, generic plotting, storage and the dependency computation context. `src/ehk` contains microscopic observables, opinion-model geometry and model-specific visualizations. Neither package imports experiment modules.

`experiments/paper` owns the original microscopic paper workflows, including `experiments/paper/mechanism_cases/social_force_probe`.

The next paper will use `src/theory` for mathematical models and `experiments/paper_theory` for experiments, analysis and figures. These directories are reserved; this cleanup does not implement new research.

Outputs belong in `outputs/`. Input datasets are listed in a small manifest inside the relevant experiment. Paths use `common.settings.register_path`.

The external `social-media-models` and `social-media-mesoscopic-models` runtimes are maintained separately. Their solvers are outside this cleanup.
