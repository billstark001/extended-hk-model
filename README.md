# Extended Hegselmann-Krause Model

## Usage

- `/w_scenarios::all_scenarios` is to define a scenario, the three parameters are:
  - The environment provider (which loads the network and initializes the agent opinions)
    - As an example . See `/env/random::RandomNetworkProvider`
  - Parameters of the HK model
  - Parameters of the simulation
- The folder `/run` stores the results of the simulation (at a certain point in time) and can be loaded with pickle
  - See `main` and `w_fig` for the script that creates the diagram
- After setting parameters, run `w_main` to start the simulation
- If an error occurs, restart `w_main`. The simulation should continue based on the data in `/run`.
