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
        for plan in plans:

            # Evaluate objective and constraints with the current plan
            print(cargoShipMO.objectives[0])
            print(cargoShipMO.objectives[1])

            print(f'plan: {len(plan)}')
            objective_values = [] * 2
            constraint_violations = [] * 2
            for i in range(2):
                print(i)
                objective_values[i] = cargoShipMO.objectives[i].evaluator(plan)
                constraint_violations[i] = cargoShipMO.constraints[i].evaluator(plan, objective_values[i])

            # Compute fitness
            constraint_violations = np.array(constraint_violations) / len(plan)
            fitness = sum(objective_values) + sum(constraint_violations)
            print(f'Fitness: {fitness}')
            plans_f.append(fitness)

        return plans_f


    ################################
    ### RECOMBINATION ALGORITHMS ###
    ################################
    # ________________ Full Recombination ________________ #
    def full_recombination(self, plan) -> List[np.ndarray]:
        new_plans = [] * self.lambda_
        for _ in range(self.lambda_):
            perm = np.random.permutation(plan)
            new_plans.append(perm)
        return new_plans

    # ________________ Single Recombination ________________ #
    def single_recombination(self, plan, num_containers) -> List[np.ndarray]:
        new_plans = [] * self.lambda_
        for _ in range(self.lambda_):
            # Select single recombination
            i = np.random.randint(num_containers)
            j = np.random.randint(num_containers)
            while i == j:
                j = np.random.randint(num_containers)
            
            # bubble method
            perm = bubble(plan, i, j)
            new_plans.append(perm)
        return new_plans

    # ________________ First Axis Recombination ________________ #
    def fisrt_axis_recombination(self, plan, num_containers) -> List[np.ndarray]:
        plan = plan.reshape((8,3,4))
        new_plans = [] * self.lambda_
        for _ in range(self.lambda_):
            perm = np.random.permutation(plan)
            new_plans.append(perm.reshape(num_containers))
        return new_plans


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
        plans = self.initialize_population(cargo_ship, num_containers)
        plans_f = self.evaluate_population(cargoShipMO, plans)

        # Genetic Algorithm: Optimization Loop
        while (f_opt > OPTIMUM) and self.budget > 0:

            # ____________________ Recombination ____________________ #
            print('Recombination...')
            plan = np.random.choice(plans)

            # Select the type and create new plans (offsprings of size lambda_)
            if self.recomb_type == FULL_RECOMBINATION:
                new_plans = self.full_recombination(plan)
            elif self.recomb_type == SINGLE_RECOMBINATION:
                new_plans = self.single_recombination(plan, num_containers)
            elif self.recomb_type == FIRST_AXIS_RECOMBINATION:
                new_plans = self.fisrt_axis_recombination(plan, num_containers)

            # ____________________ Evaluate the plans (population) ____________________ #
            print('Evaluation...')
            new_plans_f = self.evaluate_population(cargoShipMO, new_plans)

            # Take out budget
            self.budget = self.budget - self.lambda_

            # Select best
            opt_index = np.argmax(new_plans_f)
            f_opt = new_plans_f[opt_index]
            optimal = new_plans[opt_index]

            # Stoop when budget reached or optimum is found
            if (f_opt == OPTIMUM) or (self.budget <= 0): break

            # ____________________ Seletion ____________________ #
            print('Selection...')
            if (self.lambda_ / self.mu_) > 5:
                # Only consider new plans (offsprings)
                plans, plans_f = self.selection(new_plans, new_plans_f)
            else:
                # Consider plans and new plans (parents and offsprings)
                allPlans = np.concatenate((plans, new_plans))
                allPlans_f = np.concatenate((plans_f, new_plans_f))
                plans, plans_f = self.selection(allPlans, allPlans_f)
        print('\nFinished!!')
        return optimal, f_opt