# Polynomial-Trajectory-Generator

## File Descriptions

- `ptg.py` - *The primary code for generating a polynomial trajectory for some constraints. 
  This is also where weights are assigned to cost functions. 
  Adjusting these weights (and possibly adding new cost functions), can have a big effect on vehicle behavior.*
- `cost_functions.py` - *This file contains many cost functions which are used in ptg.py when selecting the best trajectory. 
  Some cost functions aren't yet implemented...*
- `evaluate_ptg.py` - *This file sets a start state, goal, and traffic conditions and runs the PTG code.
  Feel free to modify the goal, add traffic, etc... to test your vehicle's trajectory generation ability.*
- `constants.py` - *constants like speed limit, vehicle size, etc...*
- `helpers.py` - *helper functions used by other files.*
