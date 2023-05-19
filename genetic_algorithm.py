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
FIRST_AXIS_RECOMBINATION = 2

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

def fitness_sorting(gen, lonF, latF):
    '''
    Sort nextgen and nextgen_f from the small to big fitness value
    '''
    plans = gen
    plans_lonF = lonF
    plans_latF = latF
    change = True
    while(change):
        change = False
        for i in range(len(plans)-1):
            # Compute averages
            avg = (plans_lonF[i] + plans_latF[i]) / 2
            next_avg = (plans_lonF[i+1] + plans_latF[i+1]) / 2

            # Bubble sort
            if avg > next_avg:
                plans = bubble(plans, i, i+1)
                plans_lonF = bubble(plans_lonF, i, i+1)
                plans_latF = bubble(plans_latF, i, i+1)

                # aux1 = plans[i]
                # plans[i] = plans[i+1]
                # plans[i+1] = aux1

                # aux2 = plans_lonF[i]
                # plans_lonF[i] = plans_lonF[i+1]
                # plans_lonF[i+1] = aux2

                # aux3 = plans_latF[i]
                # plans_latF[i] = plans_latF[i+1]
                # plans_latF[i+1] = aux3

                change = True
    
    return np.array(plans), np.array(plans_lonF), np.array(plans_latF)


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
        self.opt_lonF = None
        self.opt_latF = None

    ####################
    ### MISCELANIOUS ###
    ####################
    def update_plans(self, plans, plans_lonF, plans_latF):
        # Sort by average fitness
        plans, plans_lonF, plans_latF = fitness_sorting(plans, plans_lonF, plans_latF)

        # Update the optimal population if needed
        for i in range(len(plans)):
            opt_avg = (self.opt_lonF + self.opt_latF) / 2
            for j in range(self.mu_):
                avg = (plans_lonF[i] + plans_latF[i]) / 2
                if avg < opt_avg[j]:
                    self.opt_plans = add_and_shift(self.opt_plans, plans[i], j)
                    self.opt_lonF = add_and_shift(self.opt_lonF, plans_lonF[i], j)
                    self.opt_latF = add_and_shift(self.opt_latF, plans_latF[i], j)
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
        plans_lonF = []
        plans_latF = []
        for plan in plans:

            # Evaluate objective and constraints with the current plan
            objective_values = [None] * 2
            constraint_violations = [None] * 2
            for i in range(2):
                objective_values[i] = cargoShipMO.objectives[i].evaluator(plan)
                constraint_violations[i] = cargoShipMO.constraints[i].evaluator(plan, objective_values[i])
            
            # Compute fitness AND Check if valid solution
            if constraint_violations[1] > 0:
                lat_fitness = lon_fitness = np.inf # Not valid solution
            else:
                norm_violations = constraint_violations[0] / len(plan)
                lon_fitness = objective_values[0] + norm_violations
                lat_fitness = objective_values[1] + norm_violations

            # Append the fitness values
            plans_lonF.append(lon_fitness)
            plans_latF.append(lat_fitness)

            # Take out budget
            if (self.budget % 100000 == 0) and (self.budget is not self.initial_budget):
                print(f'\n### Budget -->{self.initial_budget - self.budget}/{self.initial_budget} ###')
            self.budget -= 1

        return plans_lonF, plans_latF


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

    # ________________ First Axis Recombination ________________ #
    def fisrt_axis_recombination(self, plan, num_containers) -> List[np.ndarray]:
        plan = plan.reshape((8,3,4))
        newPlans = [] * self.lambda_
        for _ in range(self.lambda_):
            perm = np.random.permutation(plan)
            newPlans.append(perm.reshape(num_containers))

        return newPlans


    ############################
    ### SELECTION ALGORITHMS ###
    ############################
    # _______________________ Selection _______________________ #
    def selection(self, plans, lonF, latF):
        considering_plans, considering_lonF, considering_latF = fitness_sorting(plans, lonF, latF)
        next_plans = considering_plans[:self.mu_]
        next_lonF = considering_lonF[:self.mu_]
        next_latF = considering_latF[:self.mu_]
        
        return next_plans, next_lonF, next_latF

    #########################
    ### GENETIC ALGORITHM ###
    #########################
    def geneticAlgorithm(self, cargo_ship, cargoShipMO, num_containers):

        # ____________________ Initiaise the population ____________________ #
        print('Starting...')
        plans = self.initialize_population(cargo_ship, num_containers)
        plans_lonF, plans_latF = self.evaluate_population(cargoShipMO, plans)

        # Initialise optimal plans lenngth mu_
        self.opt_plans, self.opt_lonF, self.opt_latF = fitness_sorting(plans, plans_lonF, plans_latF)

        # Genetic Algorithm: Optimization Loop
        while self.budget > 0:

            # ____________________ Recombination ____________________ #
            # print(f'Recombination... (Type -> {self.recomb_type})')
            plan = random.SystemRandom().choice(self.opt_plans)
            # print(f'Plan selected for recombination is None? {plan.shape}')
            # Select the type and create new plans (offsprings of size lambda_)
            if self.recomb_type == FULL_RECOMBINATION:
                # print('Full recombination')
                newPlans = self.full_recombination(plan)
            elif self.recomb_type == SINGLE_RECOMBINATION:
                # print('Single recombination')
                newPlans = self.single_recombination(plan, num_containers)
            elif self.recomb_type == FIRST_AXIS_RECOMBINATION:
                # print('First Axis recombination')
                newPlans = self.fisrt_axis_recombination(plan, num_containers)

            # ____________________ Evaluate the plans (population) ____________________ #
            # print('Evaluation...')
            newPlans_lonF, newPlans_latF = self.evaluate_population(cargoShipMO, newPlans)

            # Select best ones and add to optimal population if necessary
            self.update_plans(newPlans, newPlans_lonF, newPlans_latF)

            # Stoop when budget reached
            if (self.budget <= 0): break

            # ____________________ Seletion ____________________ #
            # print('Selection...')
            if (self.lambda_ / self.mu_) > 5:
                # print('Selection coma')
                # Only consider new plans (offsprings)
                plans, plans_lonF, plans_latF = self.selection(newPlans, newPlans_lonF, newPlans_latF)
            else:
                # Consider plans and new plans (parents and offsprings)
                # print('Selction plus')
                allPlans = np.concatenate((plans, newPlans))
                allPlans_lonF = np.concatenate((plans_lonF, newPlans_lonF))
                allPlans_latF = np.concatenate((plans_latF, newPlans_latF))
                plans, plans_lonF, plans_latF = self.selection(allPlans, allPlans_lonF, allPlans_latF)

        print('\nFinished!!')
        print('Budget reached')

        return self.opt_plans, self.opt_lonF, self.opt_latF