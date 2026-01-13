# Extended Hegselmann-Krause Model for Social Media Echo Chamber Dynamics

This repository contains the simulation code and analysis scripts for the research paper:
**Segregation Before Polarization: How Recommendation Strategies Shape Echo Chamber Pathways**

## Overview

This project implements an extended discrete Bounded Confidence Model (BCM) based on the Hegselmann-Krause (HK) model to investigate how different recommendation systems influence echo chamber formation on social media platforms. The model incorporates both content-based and link-based recommendation mechanisms within a dynamic social network framework.

### Key Research Questions

1. How do content-based recommendation systems influence the evolution pathways and final state of echo chambers?
2. What are the societal meanings of these pathways from individual- and collective-level perspectives across different algorithms and societies?

### Main Findings

[TODO]

## Repository Structure

```text
├── ehk-model/              # Core simulation engine (Go implementation)
│   ├── model/              # Agent-based model components
│   │   ├── agent.go        # HK agent implementation
│   │   ├── model.go        # Main model logic
│   │   ├── recsys.go       # Recommendation system interface
│   │   └── tweet.go        # Tweet and information sharing
│   ├── recsys/             # Recommendation system implementations
│   │   ├── opinion.go      # Content-based recommendation
│   │   ├── structure.go    # Link-based recommendation
│   │   ├── random.go       # Baseline random recommendation
│   │   └── mix.go          # Hybrid recommendation systems
│   ├── simulation/         # Simulation management and serialization
│   │   ├── scenario.go     # Scenario execution
│   │   ├── event-db.go     # Event logging database
│   │   └── acc-mod-state.go # Accumulative state tracking
│   └── utils/              # Network utilities and graph operations
├── works/                  # Experiment orchestration (Python)
│   ├── simulate/           # Simulation running scripts
│   ├── stat/               # Statistical analysis scripts
│   ├── plot/               # Visualization scripts
│   └── config.py           # Experiment configurations
├── result_interp/          # Result interpretation utilities
│   ├── parse_events_db.py  # Event database parser
│   └── record.py           # Data record structures
├── stats/                  # Statistical analysis modules
│   ├── distance.py         # Opinion distance metrics
│   └── distance_c.py       # Optimized distance calculations
├── utils/                  # General utilities
│   ├── plot.py             # Plotting utilities
│   └── sqlalchemy.py       # Database utilities
└── scripts/                # Maintenance scripts
    ├── merge_db.py         # Database merging
    └── clear_db.py         # Database cleanup
```

## Model Architecture

### Agent-Based Model Components

The model implements a discrete-time agent-based simulation where:

- **Agents** represent social media users with continuous opinions in [-1, 1]
- **Network** is a directed graph representing follow relationships
- **Tweets** carry opinion information and can be original posts or retweets
- **Recommendation Systems** suggest content to users based on different strategies

### Agent Behavior

Each agent follows these rules at each simulation step:

1. **View Content**: Observe tweets from followed neighbors and recommended content
2. **Opinion Update**: Update opinion based on concordant content (within tolerance threshold ε)
   - Opinion change: Δo = μ × (average of concordant opinions - current opinion)
   - μ: decay/influence parameter
3. **Post/Retweet**: With probability ρ, retweet concordant content; otherwise post new tweet
4. **Rewire**: With probability γ, unfollow discordant neighbor and follow concordant recommended user

### Recommendation Systems

Three main recommendation strategies are implemented:

1. **Random Recommendation** (`Random`): Baseline strategy selecting users randomly
2. **Structure-Based Recommendation** (`StructureM9`): Link-based, recommending based on network proximity
3. **Opinion-Based Recommendation** (`OpinionM9`): Content-based, recommending based on opinion similarity
   - Maintains sorted index of tweets by opinion
   - Recommends content with minimal opinion distance
   - Supports historical tweet retention (parameter: `TweetRetainCount`)

### Key Parameters

- **Tolerance (ε)**: Opinion difference threshold for concordance (default: 0.45)
- **Decay/Influence (μ)**: Opinion update rate (default: 0.05)
- **Rewiring Rate (γ)**: Probability of network rewiring (default: 0.05)
- **Retweet Rate (ρ)**: Probability of retweeting vs. posting (default: 0.3)
- **RecsysCount**: Number of recommendations per agent per step (default: 10)
- **TweetRetainCount**: Number of historical tweets retained (0-6)

## Installation and Setup

### Prerequisites

- **Go** 1.20 or higher (for simulation engine)
- **Python** 3.8 or higher (for orchestration and analysis)
- **Required Python packages**: See `requirements.txt`

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/BillStark001/extended-hk-model.git
cd extended-hk-model
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Build Go simulation engine**

```bash
cd ehk-model
go build -o main main.go
cd ..
```

4. **Configure workspace paths**

Create a `sim_ws.json` file defining workspace directories:

```json
{
  "gradation": "/path/to/gradation/workspace",
  "epsilon": "/path/to/epsilon/workspace",
  "replicate": "/path/to/replicate/workspace",
  "mech": "/path/to/mechanism/workspace"
}
```

5. **Set environment variables**

Create a `.env` file:

```bash
SIMULATION_STAT_DIR=/path/to/statistics/output
SIMULATION_INSTANCE_NAME=gradation  # or epsilon, replicate, mech
STAT_THREAD_COUNT=6
```

## Running Simulations

### Quick Start Example

```bash
# Run a single simulation with specific parameters
./ehk-model/main /path/to/output '{"UniqueName":"test","Tolerance":0.45,"Decay":0.05,"RewiringRate":0.05,"RetweetRate":0.3,"RecsysFactoryType":"OpinionM9","RecsysCount":10,"TweetRetainCount":3,"MaxSimulationStep":15000}'
```

### Batch Simulations

The repository includes pre-configured experiment scenarios:

1. **Gradation Study** (Parameter sweep)

```python
from works.simulate.gradation import run_gradation
run_gradation()  # Sweeps decay, rewiring, retweet rates
```

2. **Epsilon Study** (Tolerance threshold analysis)

```python
from works.simulate.epsilon import run_epsilon
run_epsilon()  # Varies tolerance from 0.05 to 1.0
```

3. **Replication Study** (Statistical validation)

```python
from works.simulate.replicate import run_replicate
run_replicate()  # Runs 100 replications per condition
```

4. **Mechanism Study** (Pathway analysis)

```python
from works.simulate.mech import run_mech
run_mech()  # Analyzes specific mechanistic phases
```

### Simulation Output

Each simulation produces:

- **Graph snapshots** (`graph-{step}.msgpack`): Network structure at key steps
- **Accumulative state** (`acc-state-{timestamp}.lz4`): Compressed time-series data
- **Event database** (`events.db`): SQLite database with detailed agent events
- **Final state** (`finished-{timestamp}.msgpack`): Complete end state

## Data Analysis

### Event Database Schema

The event database tracks three event types:

1. **Rewiring Events**: Network structure changes
2. **Tweet Events**: Posts and retweets with opinion values
3. **ViewTweets Events**: Detailed content exposure records

### Analysis Scripts

```python
# Load and analyze simulation results
from result_interp.parse_events_db import load_events_db, get_events_by_step_range

db = load_events_db('path/to/events.db')
events = get_events_by_step_range(db, 0, 1000, type_="Rewiring")

# Calculate opinion polarization metrics
from stats.distance import calculate_polarization
polarization = calculate_polarization(opinions)
```

### Visualization

```python
from utils.plot import plot_opinion_evolution
plot_opinion_evolution(opinion_time_series, save_path='evolution.png')
```

## Experimental Configurations

The `works/config.py` file defines all experimental scenarios:

- **all_scenarios_grad**: 10 simulations × 8 rewiring rates × 8 decay rates × 4 retweet rates × 4 recommendation systems
- **all_scenarios_eps**: 100 simulations × 16 tolerance values
- **all_scenarios_rep**: 100 replications × 2 recommendation systems
- **all_scenarios_mech**: 9 mechanism-focused configurations

## Extending the Model

### Adding Custom Recommendation Systems

Follow these steps to create a custom recommendation system:

**Step 1:** Implement the `HKModelRecommendationSystem` interface in Go:

```go
type CustomRecsys struct {
    model.BaseRecommendationSystem
    Model *model.HKModel
    // Your custom fields
}

func (r *CustomRecsys) Recommend(agent *model.HKAgent, neighborIDs map[int64]bool, count int) []*model.TweetRecord {
    // Your recommendation logic
}
```

**Step 2:** Register in `simulation/scenario-metadata.go`:

```go
RECSYS_FACTORY["CustomType"] = func(m *model.HKModel) model.HKModelRecommendationSystem {
    return NewCustomRecsys(m)
}
```

### Modifying Agent Behavior

Edit `ehk-model/model/agent.go`, particularly the `HKAgentStep` function which implements the core agent decision-making logic.

## Citation

If you use this code in your research, please cite:

`[TODO BibTeX Reference]`

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

This research investigates the critical role of algorithmic recommendations in fostering polarization on social media platforms. The findings provide theoretical frameworks for stage-dependent interventions, suggesting platforms could dynamically adjust algorithms to mitigate polarization without resorting to censorship.
