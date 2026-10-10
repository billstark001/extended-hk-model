# Repository architecture

Reusable microscopic code lives in `src/ehk`; the original paper's workflows live in `experiments/paper`. Its mechanism probe lives in `experiments/paper/mechanism_cases/social_force_probe`.

The next paper will use `src/theory` for mathematical models and `experiments/paper_theory` for experiments, analysis and figures. These directories are reserved; this cleanup does not implement new research.

Outputs belong in `outputs/`. Input datasets are listed in a small manifest inside the relevant experiment. Paths use `ehk.common.settings.register_path`.

The external `social-media-models` and `social-media-mesoscopic-models` runtimes are maintained separately. Their solvers are outside this cleanup.
