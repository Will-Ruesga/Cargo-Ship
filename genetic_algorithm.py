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
def fitness_sorting(next, next_f):
    '''
    Sort nextgen and nextgen_f from the best to worst fitness value
    '''
    nextgen = next
    nextgen_f = next_f
    change = True
    while(change):
        change = False
        for i in range(len(nextgen_f)-1):
            if nextgen_f[i] < nextgen_f[i+1]:
                aux1 = nextgen[i]
                nextgen[i] = nextgen[i+1]
                nextgen[i+1] = aux1

                aux3 = nextgen_f[i]
                nextgen_f[i] = nextgen_f[i+1]
                nextgen_f[i+1] = aux3

                change = True
    
    return nextgen, nextgen_f

def bubble(array, index1, index2):
    '''
    Swap two array values given the indexes
    '''
    new_arr = array
    aux = new_arr[index1]
    new_arr[index1] = new_arr[index2]
    new_arr[index2] = aux
    return new_arr

class GeneticAlgorithm():
    def __init__(self, mu_, lambda_, budget, recomb_type):
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.budget = budget
        self.initial_budget = budget
        self.recomb_type = recomb_type
        
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
        plans_f = []
        plans_v = []
        status = False
        for plan in plans:
            valid = True

            # Evaluate objective and constraints with the current plan
            objective_values = [None] * 2
            constraint_violations = [None] * 2
            for i in range(2):
                objective_values[i] = cargoShipMO.objectives[i].evaluator(plan)
                constraint_violations[i] = cargoShipMO.constraints[i].evaluator(plan, objective_values[i])
            
            # Check if it is a valid solution
            if constraint_violations[1] > 0:
                valid = False
            
            # Compute fitness
            constraint_violations = np.array(constraint_violations) / len(plan)
            fitness = sum(objective_values) + sum(constraint_violations)
            # print(f'Fitness: {fitness}')
            plans_f.append(fitness)
            plans_v.append(valid)

            # Take out budget
            if self.budget % 100000 == 0:
                print(f'\n### Budget -->{self.budget}/{self.initial_budget} ###')
                status = True
            self.budget -= 1

        return plans_f, plans_v, status


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
    def selection(self, plans: List[np.ndarray], fitness_values: List[float]):
        considering_plans, considering_fitness = fitness_sorting(plans, fitness_values)
        next_plans = considering_plans[:self.mu_]
        next_fitnes = considering_fitness[:self.mu_]
        
        return next_plans, next_fitnes

    #########################
    ### GENETIC ALGORITHM ###
    #########################
    def geneticAlgorithm(self, cargo_ship, cargoShipMO, num_containers):

        # ____________________ Initiaise the population ____________________ #
        print('Initialise...')
        f_opt = np.inf
        plans = self.initialize_population(cargo_ship, num_containers)
        plans_f, plans_v, _ = self.evaluate_population(cargoShipMO, plans)
        
        # Select best
        id_min = np.argmin(plans_f)
        if (f_opt > plans_f[id_min]) and (plans_v[id_min] == True):
            opt_index = id_min
            f_opt = plans_f[opt_index]
            optimal = plans[opt_index]


        # Genetic Algorithm: Optimization Loop
        while (f_opt > OPTIMUM) and self.budget > 0:

            # ____________________ Recombination ____________________ #
            # print(f'Recombination... (Type -> {self.recomb_type})')
            plan = random.SystemRandom().choice(plans)
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
            newPlans_f, newPlans_v, status = self.evaluate_population(cargoShipMO, newPlans)
            if status:
                print(f'Status report:')
                print(f'  Current best fitness -> {f_opt}')

            # Select best
            id_min = np.argmin(newPlans_f)
            if (f_opt > newPlans_f[id_min]) and (newPlans_v[id_min] == True):
                opt_index = id_min
                f_opt = newPlans_f[opt_index]
                optimal = newPlans[opt_index]

            # Stoop when budget reached or optimum is found
            # print(f'Current Best Plan: {f_opt}')
            if (f_opt == OPTIMUM) or (self.budget <= 0): break

            # ____________________ Seletion ____________________ #
            # print('Selection...')
            if (self.lambda_ / self.mu_) > 5:
                # print('Selection coma')
                # Only consider new plans (offsprings)
                plans, plans_f = self.selection(newPlans, newPlans_f)
            else:
                # Consider plans and new plans (parents and offsprings)
                # print('Selction plus')
                allPlans = np.concatenate((plans, newPlans))
                allPlans_f = np.concatenate((plans_f, newPlans_f))
                plans, plans_f = self.selection(allPlans, allPlans_f)

        print('\nFinished!!')
        if f_opt == OPTIMUM:
            print('Optimum reached')
        else:
            print('Budget reached')
        return optimal, f_opt