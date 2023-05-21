###############
### Imports ###
###############

# General
import random
import numpy as np

# Genetic Algorithm
from typing import List, Tuple

FULL_RECOMBINATION = 0
SINGLE_RECOMBINATION = 1
FULL_AXIS_RECOMBINATION = 2
SINGLE_AXIS_RECOMBINATION = 3
rng = np.random.default_rng()

OPTIMUM = 0

####################
### MISCELANIOUS ###
####################
def bubble(array, index1, index2):
    '''
    Swap two array values given the indexes
    '''
    new_arr = array
    aux = new_arr[index1]
    new_arr[index1] = new_arr[index2]
    new_arr[index2] = aux
    return new_arr

def add_and_shift(array, element, to):
    '''
    Adds new element to array and shifts backwards the rest
    '''
    arr = array.copy()
    fr = len(arr) - 1
    if fr >= arr.size or to >= arr.size:
        return None
    if fr > to:
        arr[to+1:fr+1] = arr[to:fr]
    else:
        arr[fr:to] = arr[fr+1:to+1]
    arr[to] = element
    return arr

def fitness_sorting(gen, fit1, fit2):
    '''
    Sort nextgen and nextgen_f from the small to big fitness value
    '''
    plans = gen
    plans_fit1 = fit1
    plans_fit2 = fit2
    change = True
    while(change):
        change = False
        for i in range(len(plans)-1):

            # Compute averages
            avg = (plans_fit1[i] + plans_fit2[i]) / 2
            next_avg = (plans_fit1[i+1] + plans_fit2[i+1]) / 2

            # Bubble sort
            if avg > next_avg:
                plans = bubble(plans, i, i+1)
                plans_fit1 = bubble(plans_fit1, i, i+1)
                plans_fit2 = bubble(plans_fit2, i, i+1)

                change = True
    
    return np.array(plans), np.array(plans_fit1), np.array(plans_fit2)

def fitness_sorting1(gen, fit1, fit2):
    '''
    Sort nextgen and nextgen_f from the small to big fitness value
    '''
    plans = gen
    plans_fit1 = fit1
    plans_fit2 = fit2
    change = True
    while(change):
        change = False
        for i in range(len(plans_fit1)-1):

            # Bubble sort
            if plans_fit1[i] > plans_fit1[i+1]:
                plans = bubble(plans, i, i+1)
                plans_fit1 = bubble(plans_fit1, i, i+1)
                plans_fit2 = bubble(plans_fit2, i, i+1)

                change = True
    
    return np.array(plans), np.array(plans_fit1), np.array(plans_fit2)

def identify_pareto(plans, fit1, fit2):
    pareto_plans = []
    pareto_fit2 = []
    pareto_fit1 = []
    pareto_plans.append(plans[0])
    pareto_fit1.append(fit1[0])
    pareto_fit2.append(fit2[0])
    for i in range(1, len(fit2)):
        if fit2[i] < pareto_fit2[-1]:
            pareto_plans.append(plans[i])
            pareto_fit1.append(fit1[i])
            pareto_fit2.append(fit2[i])

    return np.array(pareto_plans), np.array(pareto_fit1), np.array(pareto_fit2)


###################################################################################################################################
# ------------------------------------------------------ GENETIC ALGORITHM ------------------------------------------------------ #
###################################################################################################################################
class GeneticAlgorithm():
    def __init__(self, mu_, lambda_, budget, recomb_type):
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.budget = budget
        self.initial_budget = budget
        self.recomb_type = recomb_type
        self.opt_plans = None
        self.opt_fit1 = None
        self.opt_fit2 = None

    ####################
    ### MISCELANIOUS ###
    ####################
    def update_plans(self, plans, plans_fit1, plans_fit2):
        # Sort by average fitness
        plans, plans_fit1, plans_fit2 = fitness_sorting(plans, plans_fit1, plans_fit2)

        # Update the optimal population if needed
        for i in range(len(plans)):
            opt_avg = (self.opt_fit1 + self.opt_fit2) / 2
            for j in range(self.mu_):
                avg = (plans_fit1[i] + plans_fit2[i]) / 2
                if avg < opt_avg[j]:
                    self.opt_plans = add_and_shift(self.opt_plans, plans[i], j)
                    self.opt_fit1 = add_and_shift(self.opt_fit1, plans_fit1[i], j)
                    self.opt_fit2 = add_and_shift(self.opt_fit2, plans_fit2[i], j)
                    i += 1


    #############################
    ### INITIALISE POPULATION ###
    #############################
    def initialize_population(self, cargo_ship, num_containers) -> List[np.ndarray]:
        plans = []
        flt_cargo_ship = cargo_ship.reshape(num_containers)
        for _ in range(self.mu_):
            unload_plan = np.random.permutation(flt_cargo_ship)
            plans.append(unload_plan)
        return plans


    ###########################
    ### EVALUATE POPULATION ###
    ###########################
    def evaluate_population(self, cargoShipMO, plans) -> List[float]:
        plans_gravF = []
        plans_unloadF = []
        for plan in plans:

            # Evaluate objective and constraints with the current plan
            objective_values = [None] * 2
            constraint_violations = [None] * 2
            for i in range(2):
                objective_values[i] = cargoShipMO.objectives[i].evaluator(plan)
                constraint_violations[i] = cargoShipMO.constraints[i].evaluator(plan, objective_values[i])
            
            # Compute fitness AND Check if valid solution
            if constraint_violations[1] > 0:
                grav_fitness = unloading_fitness = np.inf # Not valid solution
            else:
                unloading_fitness = constraint_violations[0] / len(plan)
                grav_fitness = np.sqrt(objective_values[0]**2 + objective_values[1]**2)

            # Append the fitness values
            plans_gravF.append(grav_fitness)
            plans_unloadF.append(unloading_fitness)

            # Take out budget
            if (self.budget % (self.initial_budget/10) == 0) and (self.budget is not self.initial_budget):
                print(f'\n### Budget -->{self.initial_budget - self.budget}/{self.initial_budget} ###')
            self.budget -= 1

        return plans_gravF, plans_unloadF


    ################################
    ### RECOMBINATION ALGORITHMS ###
    ################################
    # ________________ Full Recombination ________________ #
    def full_recombination(self, plan) -> List[np.ndarray]:
        newPlans = [] * self.lambda_
        for _ in range(self.lambda_):
            perm = np.random.permutation(plan)
            newPlans.append(perm)

        return newPlans

    # ________________ Single Recombination ________________ #
    def single_recombination(self, plan, num_containers) -> List[np.ndarray]:
        newPlans = [] * self.lambda_
        for _ in range(self.lambda_):

            # Select single recombination
            i = np.random.randint(num_containers)
            j = np.random.randint(num_containers)
            while i == j:
                j = np.random.randint(num_containers)
            
            # bubble method
            perm = bubble(plan, i, j)
            newPlans.append(perm)

        return newPlans

    # ________________ Full Axis Recombination ________________ #
    def full_axis_recombination(self, plan, num_containers, axis) -> List[np.ndarray]:
        plan = plan.reshape((8,3,4))
        newPlans = [] * self.lambda_
        for _ in range(self.lambda_):
            perm = rng.permutation(plan, axis=axis)
            newPlans.append(perm.reshape(num_containers))

        return newPlans
    
    # ________________ Single Axis Recombination ________________ #
    def single_axis_recombination(self, plan, num_containers, axis) -> List[np.ndarray]:
        dims = [8,3,4]
        plan = plan.reshape((8,3,4))
        newPlans = [] * self.lambda_
        for _ in range(self.lambda_):
            perm = plan.copy()
            if axis == 0:
                perm[:, random.randint(0, dims[1]-1), random.randint(0, dims[2]-1)] = perm[:, random.randint(0, dims[1]-1), random.randint(0, dims[2]-1)]
            elif axis == 1:
                perm[random.randint(0, dims[0]-1), :, random.randint(0, dims[2]-1)] = perm[random.randint(0, dims[0]-1), :, random.randint(0, dims[2]-1)]
            elif axis == 2:
                perm[random.randint(0, dims[0]-1), random.randint(0, dims[1]-1), :] = perm[random.randint(0, dims[0]-1), random.randint(0, dims[1]-1), :]
            newPlans.append(perm.reshape(num_containers))

        return newPlans


    ############################
    ### SELECTION ALGORITHMS ###
    ############################
    # _______________________ Selection _______________________ #
    def selection(self, plans, fit1, fit2):
        considering_plans, considering_fit1, considering_fit2 = fitness_sorting(plans, fit1, fit2)
        next_plans = considering_plans[:self.mu_]
        next_fit1 = considering_fit1[:self.mu_]
        next_fit2 = considering_fit2[:self.mu_]
        
        return next_plans, next_fit1, next_fit2

    #########################
    ### GENETIC ALGORITHM ###
    #########################
    def geneticAlgorithm(self, cargo_ship, cargoShipMO, num_containers):

        # ____________________ Initiaise the population ____________________ #
        print('Starting...')
        plans = self.initialize_population(cargo_ship, num_containers)
        plans_fit1, plans_fit2 = self.evaluate_population(cargoShipMO, plans)

        allPlans = plans
        allPlans_fit1 = plans_fit1
        allPlans_fit2 = plans_fit2

        # Initialise optimal plans lenngth mu_
        self.opt_plans, self.opt_fit1, self.opt_fit2 = fitness_sorting(plans, plans_fit1, plans_fit2)

        # Genetic Algorithm: Optimization Loop
        while self.budget > 0:

            # ____________________ Recombination ____________________ #
            plan = random.SystemRandom().choice(self.opt_plans)

            # Select the type and create new plans (offsprings of size lambda_)
            if self.recomb_type == FULL_RECOMBINATION:
                newPlans = self.full_recombination(plan)
            elif self.recomb_type == SINGLE_RECOMBINATION:
                newPlans = self.single_recombination(plan, num_containers)
            elif self.recomb_type == FULL_AXIS_RECOMBINATION:
                newPlans = self.full_axis_recombination(plan, num_containers, random.randint(0,2))
            elif self.recomb_type == SINGLE_AXIS_RECOMBINATION:
                newPlans = self.single_axis_recombination(plan, num_containers, random.randint(0,2))

            # ____________________ Evaluate the plans (population) ____________________ #
            newPlans_fit1, newPlans_fit2 = self.evaluate_population(cargoShipMO, newPlans)

            # Select best ones and add to optimal population if necessary
            self.update_plans(newPlans, newPlans_fit1, newPlans_fit2)

            # Stop when budget reached
            if (self.budget <= 0): break

            # ____________________ Selection ____________________ #
            if (self.lambda_ / self.mu_) > 5:

                # Only consider new plans (offsprings)
                plans, plans_fit1, plans_fit2 = self.selection(newPlans, newPlans_fit1, newPlans_fit2)
            else:
                # Consider plans and new plans (parents and offsprings)
                allPlans = np.concatenate((allPlans, newPlans))
                allPlans_fit1 = np.concatenate((allPlans_fit1, newPlans_fit1))
                allPlans_fit2 = np.concatenate((allPlans_fit2, newPlans_fit2))

                newPlans = np.concatenate((plans, newPlans))
                newPlans_fit1 = np.concatenate((plans_fit1, newPlans_fit1))
                newPlans_fit2 = np.concatenate((plans_fit2, newPlans_fit2))
                plans, plans_fit1, plans_fit2 = self.selection(newPlans, newPlans_fit1, newPlans_fit2)

        pareto_plans, fit1, fit2 = fitness_sorting1(allPlans, allPlans_fit1, allPlans_fit2)
        self.opt_plans, self.opt_fit1, self.opt_fit2 = identify_pareto(pareto_plans, fit1, fit2)
        print('\nFinished!!')
        print('Budget reached')

        return self.opt_plans, self.opt_fit1, self.opt_fit2