###############
### Imports ###
###############

# General
import random
import numpy as np
import matplotlib.pyplot as plt

# Genetic Algorithm
from typing import List, Tuple
import genetic_algorithm as ga

# DESDEO
from desdeo_tools.scalarization import SimpleASF
from desdeo_problem import Variable, ScalarObjective, MOProblem, ScalarConstraint

#######################
### CONTAINER CLASS ###
#######################
class Container():
    '''
    Defines a container, with the weight attached to it
    '''
    def __init__(self, name, weight, harbor):
        self.n = name
        self.w = weight
        self.h = harbor


###########################
### OBJECTIVE FUNCTIONS ###
###########################
def longitudinal_objective(plan):
    '''
    Calculate longitudinal center of gravityand.
    container = 12.19m long x 2.44m wide x 2.59m high
    Returns the distance from the middle of the section
    '''
    total = 0
    weighted = 0
    plan = np.reshape(plan, (bays, tiers, rows))
    for bay in range(plan.shape[0]):
        for tier in range(plan.shape[1]):
            for row in range(plan.shape[2]):
                total += plan[bay, tier, row].w
                weighted += plan[bay, tier, row].w * (bay+1)

    # print(f'Longitudinal dist to center: {abs(weighted/total-4.5)}')  
    return abs(weighted/total-4.5)


def latitudinal_objective(plan):
    '''
    Calculate latitudinal center of gravity.
    Returns the distance from the middle of the section
    '''
    total = 0
    weighted = 0
    plan = np.reshape(plan, (bays, tiers, rows))
    for bay in range(plan.shape[0]):
        for tier in range(plan.shape[1]):
            for row in range(plan.shape[2]):
                total +=  plan[bay, tier, row].w
                weighted +=  plan[bay, tier, row].w * (row+1)

    # print(f'Latitudinal dist to center: {abs(weighted/total-2.5)}')
    return abs(weighted/total-2.5)


############################
### CONSTRAINT FUNCTIONS ###
############################
def unloading_constraint(plan, obj):
    '''
    How many times containers are placed in a way that BLOCKS smooth unloading
    '''
    violations = 0
    plan = np.reshape(plan, (bays, tiers, rows))

    for bay in range(plan.shape[0]):
        for tier in range(plan.shape[1] - 1): # Check only lower tiers (bottom of the cargo ship)
            for row in range(plan.shape[2]):
                if plan[bay, tier, row].h < plan[bay, tier+1, row].h:
                    violations += 1
                    
    return violations

def loading_constraint(plan, obj):
    '''
    Ensure heavier containers are not placed on top of lighter containers
    '''
    violations = 0
    plan = np.reshape(plan, (bays, tiers, rows))

    for bay in range(plan.shape[0]):
        for tier in range(plan.shape[1]):
            for row in range(plan.shape[2]-1):
                if plan[bay][tier][row].w - plan[bay, tier, row+1].w > delta_w:
                    violations += 1

    return violations


############################
### CARGO SHIP MOPROBLEM ###
############################

def initialise_cargo_ship_problem(num_containers):
    # Define objectives
    csObjectives = [ScalarObjective(name="Longitudinal center of gravity", evaluator=longitudinal_objective, maximize=False),
                    ScalarObjective(name="Latitudinal center of gravity", evaluator=latitudinal_objective, maximize=False)]

    # Define varibles
    csVariables = np.ndarray(num_containers, dtype=Variable)
    for i in range(num_containers):
        csVariables[i] = Variable(name=f"Container {i+1}", initial_value=0)

    # Define constraints
    csConstraints = [ScalarConstraint(name="Unloading constraint", n_decision_vars=num_containers, n_objective_funs=2, evaluator=unloading_constraint),
                    ScalarConstraint(name="Loading constraint", n_decision_vars=num_containers, n_objective_funs=2, evaluator=loading_constraint)]

    # Inirtialise the MO problem
    return MOProblem(csObjectives, csVariables, csConstraints)


###########################
### SOLVING THE PROBLEM ###
###########################

# -------------- Side view -------------- #
#                   Bays                    TIERS
#   0    1    2    3    4    5    6    7
# |___||___||___||___||___||___||___||___|    2
# |___||___||___||___||___||___||___||___|    1
# |___||___||___||___||___||___||___||___|    0

# -------------- Top view -------------- #
#                   Bays                    ROWS
#   0    1    2    3    4    5    6    7
#  ___  ___  ___  ___  ___  ___  ___  ___
# |___||___||___||___||___||___||___||___|    3
# |___||___||___||___||___||___||___||___|    2
# |___||___||___||___||___||___||___||___|    1
# |___||___||___||___||___||___||___||___|    0

# _____________ Initialise Cargo Ship Problem _____________ #
# Cargo Ship distribution
bays = 8
tiers = 3
rows = 4
num_containers = bays * tiers * rows

# Initialise container harbors
# Harbor stops order --> Hamburg == 0 (First) | Aarhus == 1 (Second) | Copenhagen == 2 (Third)
harbors = ["Hamburg", "Aarhus", "Copenhagen"]
destinations = np.random.randint(0, len(harbors), size=(bays, tiers, rows))
# destinations = np.random.randint(0, len(harbors), size=num_containers)

# Initialise container weigths
delta_w = 30
weights = np.random.lognormal(2.5, 0.5, size=(bays, tiers, rows))
# weights = np.random.lognormal(2.5, 0.5, size=num_containers)

# Initialise Containers
i = 0
cargo_ship = np.ndarray((bays, tiers, rows), dtype=Container)
for bay in range(bays):
    for tier in range(tiers):
        for row in range(rows):
            i =+ 1
            cargo_ship[bay, tier, row] = Container(name=f'Container {i}',
                                                   weight=weights[bay, tier, row],
                                                   harbor=destinations[bay, tier, row])
            
# Initialise Generic Algoritm
mu_ = 20
lambda_ = 40
budget = 500000
recomb_type = 0
evolAlgo = ga.GeneticAlgorithm(mu_, lambda_, budget, recomb_type)

# _____________ Set Cargo Ship Problem as MOProblem _____________ #
cargoShipMO = initialise_cargo_ship_problem(num_containers)

# _____________ Genetic Algorithm _____________ #
optPlans, optPlans_lonF, optPlans_latF = evolAlgo.geneticAlgorithm(cargo_ship, cargoShipMO, num_containers)
optPlans = optPlans.reshape(mu_, bays, tiers, rows)
print(f'Longitudinal Fitness of Optimal Population: {optPlans_lonF}')
print(f'Latitudinal Fitness of Optimal Population: {optPlans_latF}')
print(f'Final Solution: {optPlans.shape}')


# _____________ Plot Pareto Region Comparison with Random Population _____________ #
rndPlans = evolAlgo.initialize_population(cargo_ship, num_containers)
rndPlans_lonF, rndPlans_latF = evolAlgo.evaluate_population(cargoShipMO, rndPlans)

plt.figure()
plt.scatter(optPlans_lonF, optPlans_latF, c='red')
plt.scatter(rndPlans_lonF, rndPlans_latF, c='blue')
plt.title('Cargo Ship Problem - Pareto Region (red) and Random Region (blue)')
plt.xlabel('Longitudinal optimization')
plt.ylabel('Latitudinal Optimization')
plt.show()