from typing import Any, Dict, List, TypeAlias

import os
import sys
import json

import dotenv
import numpy as np


# build scenarios

decay_rate_array = rewiring_rate_array = np.array(
    [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1]
)
retweet_rate_array = np.array([0, 0.1, 0.25, 0.5])

tolerance_array = np.array(
    [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
     0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1]
)

n_sims = 12
n_sims_eps = 100
n_sims_rep = 100

rs_names = {
    "st": ("StructureM9", 0),
    "op0": ("OpinionM9", 0),
    "op2": ("OpinionM9", 2),
    "op6": ("OpinionM9", 6),
}

# all parameters


ScenarioMetadata: TypeAlias = Dict[str, Any]


def create_scenario_metadata(
    name: str,
    tolerance=0.45,
    influence=0.9,
    rewiring=0.01,
    repost_rate=0.05,
    recsys_type="Random",
    recsys_count=10,
    post_retain_count=3,
    max_sim_step=20000,
) -> ScenarioMetadata:
  ret = {
      "UniqueName": name,
      "DynamicsType": "HK",
      "HKParams": {
          "Tolerance": tolerance,
          "Influence": influence,
          "RewiringRate": rewiring,
          "RepostRate": repost_rate,
      },
      "RecsysFactoryType": recsys_type,
      "RecsysCount": recsys_count,
      "PostRetainCount": post_retain_count,
      "MaxSimulationStep": max_sim_step,
  }
  return ret


all_scenarios_grad: List[ScenarioMetadata] = []

for i_sim in range(n_sims):
  for i_rw, rw in enumerate(rewiring_rate_array):
    for i_dc, dc in enumerate(decay_rate_array):
      for i_rt, rt in enumerate(retweet_rate_array):
        for k_rs, (rs, t_retain) in rs_names.items():
          x = create_scenario_metadata(
              f"s_grad_sim{i_sim}_rw{i_rw}_dc{i_dc}_rt{i_rt}_{k_rs}",
              rewiring=rw,
              influence=dc,
              repost_rate=rt,
              recsys_type=rs,
              post_retain_count=t_retain,
          )
          all_scenarios_grad.append(x)


all_scenarios_eps: List[ScenarioMetadata] = []

for i_sim in range(n_sims_eps):
  for i_to, to in enumerate(tolerance_array):
    x = create_scenario_metadata(
        f"s_eps_sim{i_sim}_to{i_to}",
        rewiring=0.05,
        influence=0.05,
        repost_rate=0.1,
        recsys_type="Random",
        post_retain_count=3,
        tolerance=to,
        max_sim_step=20000,
    )
    all_scenarios_eps.append(x)


all_scenarios_rep: List[ScenarioMetadata] = []

for i_sim in range(n_sims_rep):
  for i_rs, rs in [("rn", "Random"), ("st", "StructureM9")]:
    x = create_scenario_metadata(
        f"s_rep_sim{i_sim}_{i_rs}",
        rewiring=0.05,
        influence=0.05,
        repost_rate=0.1,
        recsys_type=rs,
        post_retain_count=3,
        tolerance=0.45,
        max_sim_step=20000,
    )
    all_scenarios_rep.append(x)


mech_phases = [
    (7, 0, 3, "StructureM9", 0),
    (7, 0, 0, "StructureM9", 0),
    (4, 4, 0, "OpinionM9", 6),
    (2, 7, 0, "OpinionM9", 6),
    (2, 7, 0, "StructureM9", 0),
]

all_scenarios_mech: List[ScenarioMetadata] = [
    create_scenario_metadata(
        "s_mech_baseline",
        rewiring=0.05,
        influence=0.05,
        repost_rate=0,
        recsys_type="Random",
    ),
    create_scenario_metadata(
        "s_mech_baseline_pbs",
        rewiring=0.025,
        influence=0.05,
        repost_rate=0,
        recsys_type="Random",
    ),
    create_scenario_metadata(
        "s_mech_baseline_sbp",
        rewiring=0.025,
        influence=0.005,
        repost_rate=0,
        recsys_type="Random",
    ),
    create_scenario_metadata(
        "s_mech_influence",
        rewiring=0.05,
        influence=0.1,
        repost_rate=0,
        recsys_type="Random",
    ),
    create_scenario_metadata(
        "s_mech_retweet",
        rewiring=0.05,
        influence=0.05,
        repost_rate=0.1,
        recsys_type="Random",
    ),
    create_scenario_metadata(
        "s_mech_op_recsys",
        rewiring=0.05,
        influence=0.05,
        repost_rate=0,
        recsys_type="OpinionM9",
    ),
    *[
        create_scenario_metadata(
            f"s_mech_phase{i+1}",
            rewiring=rewiring_rate_array[rw],
            influence=decay_rate_array[dc],
            repost_rate=retweet_rate_array[rt],
            recsys_type=rs,
            post_retain_count=t_retain,
        )
        for i, (rw, dc, rt, rs, t_retain) in enumerate(mech_phases)
    ],
]

# assign paths

# path related

dotenv.load_dotenv()


def normalize_path(p: str):
  if p == "" or p == None:
    raise ValueError("Empty path")
  p = os.path.expandvars(p)
  p = os.path.expanduser(p)
  p = os.path.abspath(p)
  p = os.path.normpath(p)
  # if os.name == 'nt':
  #   p = p.replace('\\', '/')
  return p


def normalize_int(p: str | None, default: int = 0):
  if not p:
    return default
  try:
    return int(p.strip())
  except ValueError:
    pass  # do nothing
  return default


STAT_THREAD_COUNT = max(normalize_int(
    os.environ.get("STAT_THREAD_COUNT", None), 6), 1)

SMP_BINARY_PATH = normalize_path(
    os.environ.get("SMP_BINARY_PATH", "../social-media-models/smp")
)
SIMULATION_WS_PATH = normalize_path(
    os.environ.get("SIMULATION_WS_PATH", "./sim_ws.json")
)
SIMULATION_STAT_DIR = normalize_path(os.environ["SIMULATION_STAT_DIR"])

with open(SIMULATION_WS_PATH, "r", encoding="utf-8") as f:
  workspace_def: Dict[str, str] = json.load(f)


def get_instance_name_raw(
    name: str | None = None, default: str | None = None
) -> str | None:
  if name is None:
    name = sys.argv[1] if len(sys.argv) > 1 else None
  if not name:
    name = os.environ.get("SIMULATION_INSTANCE_NAME", None) or default
  return name


def get_instance_name(name: str | None = None, default: str | None = None) -> str:
  instance_name = get_instance_name_raw(name, default)
  if not instance_name:
    raise ValueError(f"Simulation instance name not defined")
  return instance_name


def get_workspace_dir(name: str | None = None, default: str | None = None) -> str:
  instance_name = get_instance_name(name, default)
  if not instance_name or instance_name not in workspace_def:
    raise ValueError(
        f"Workspace path for {instance_name or '<empty>'} not defined")
  return normalize_path(workspace_def[instance_name])
