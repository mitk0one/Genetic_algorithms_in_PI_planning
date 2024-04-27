"""
You are a Scrum Master and your goal is to plan the workload for your team for the next PI.
TO ADD MORE DESCRIPTION
"""


import random
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='status.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Initialization (Generate random stories)

def generate_story_details():

    # Set seed for reproducibility (optional)
    prio_imp = np.random.choice(2)
    prio_urg = np.random.choice(2)

    bv = np.round(a=(np.random.normal(loc=5, scale=3)), decimals=1)
    bv = max(0.1, min(bv, 10.0))

    compl_lvl = np.random.choice(3)

    block_risk = np.random.choice(3)

    est = np.round(a=(np.random.normal(loc=3, scale=3)), decimals=1)
    est = max(0.3, min(est, 24.0))

    return {
        "prio_imp": prio_imp,
        "prio_urg": prio_urg,
        "bv": bv,
        "compl_lvl": compl_lvl,
        "block_risk": block_risk,
        "est": est
    }


def generate_story(num_stories):
    st_dict = {}

    for i in range(1, num_stories + 1):
        story_name = f"story_{i}"
        st_dict[story_name] = generate_story_details()  # value of range should equal the total num of developers (or the developers left)

    return st_dict


def calc_mutual_prio(story_dict):

    for story in story_dict.keys():

        importance = story_dict[story]["prio_imp"]
        urgency = story_dict[story]["prio_urg"]

        if urgency:
            if importance:
                story_dict[story]["mutual_prio"] = 4.5
            else:
                story_dict[story]["mutual_prio"] = 2.5
        else:
            if importance:
                story_dict[story]["mutual_prio"] = 2
            else:
                story_dict[story]["mutual_prio"] = 1

    return story_dict



# 1. Generate random arrays


class Initialization:

    n_stories = 140

    stories_dict = generate_story(num_stories=n_stories)
    stories_dict = calc_mutual_prio(story_dict=stories_dict)

    developers = {
        "Senior": 2,
        "Mid": 3,
        "Junior": 2
    }

    init_story_sel_prob = 0.5
    total_working_days = 70

    def __init__(self, size:int):

        self.chosen_items = [1 if el >= Initialization.init_story_sel_prob else 0 for el in np.random.random(size)]
        self.fitness = -1

    def __str__(self):
        # return f"Chosen items:  {self.chosen_items}    Fitness score: {self.fitness}"
        logging.info("")

        return f"Fitness score: {self.fitness}"



# 2. Set up parameters

max_weight = 20
population_size = 5000
generations = 50
selection_thresh = 0.2
crossover_rate = 0.5
# crossover_slices = 5
mutation_rate = 0.08





# 3. Define main function

def ga():

    """
    elements is a list of objects;
    each element is a object, containing a fitness score and list of chosen items as attributes;
    """
    elements = init_population(pop_size=population_size,
                               input_length=Initialization.n_stories)
    # print(elements)
    # print(elements[0])
    # print(f"Element old fitness: {elements[0].fitness}")

    for generation in range(generations):

        print(f"Generation: {generation}")


        elements = fitness_score(class_elements=elements,
                                 stories=Initialization.stories_dict,
                                 pi_time_limit=Initialization.total_working_days,
                                 dev_pool=Initialization.developers)

        # elements = Selection.top_members_selection(members=elements, thresh=selection_thresh)
        # elements = Selection.tournament_selection(class_elements=elements, thresh=selection_thresh)
        elements = Selection.roulette_selection(class_elements=elements, thresh=selection_thresh)
        elements = Evolution.crossover(class_elements=elements, rate=crossover_rate)
        # elements = Evolution.crossover_mult_slices(members=elements, slices=crossover_slices)
        elements = Evolution.mutation(class_elements=elements, rate=mutation_rate)

        elements = fitness_score(class_elements=elements,
                                 stories=Initialization.stories_dict,
                                 pi_time_limit=Initialization.total_working_days,
                                 dev_pool=Initialization.developers)
        #
        elements = sorted(elements, key=lambda elements: elements.fitness, reverse=False)

        logging.info(f"Generation BEST Score:  {elements[-1].fitness}")
        logging.info(f"Generation BEST selection: {elements[-1].chosen_items}")
        print(f"Generation BEST Score:  {elements[-1].fitness}")



        # print('\n'.join(map(str, elements)))
        #
        # if any(element.fitness >= 100 for element in elements):
        #     print(f"Problem Solved in Generation ! {generation}")
        #     break
        #
        # print(elements)
        # print(f"Element NEW fitness: {elements[0].fitness}")
        # break


# 4. Define initialization, fitness, selection, crossover and mutation functions

def init_population(pop_size, input_length):
    """returns a list of objects each containing selected elements from items list"""
    return [Initialization(input_length) for _ in range(pop_size)]


def fitness_score(class_elements, stories, pi_time_limit, dev_pool):
    """
    Fitness score formula: business_value*mutual_priority OR (bv*mutual_prio)

    Restrictions:

    1. Total estimation time of all selected stories should not exceed the total PI working days.
    2. Level of complexity should correspond to the seniority of the devs (same as 3.)
    3. Total estimation time per level of complexity should not exceed the total capacity of the corresponding developer level
    4. If average risk for blockers is higher than 1, the total estimation of the stories would change with a % according to the avg value.
    5. 80% of the total capacity of the devs should be used (per Seniority level)

    """

    iteration = -1

    # for each list of stories (each chromosome)
    for class_element in class_elements:

        # print()
        # print()
        iteration += 1

        # print(f"iteration: {iteration}")

        # keys should be "all, high_compl, medium_compl, low_compl"; values should be lists
        est_times = {
            "all": [],
            "high_compl": [],
            "medium_compl": [],
            "low_compl": []
        }

        # keys should be "senior, mid, junior"
        tot_capacity = {
            "senior": [],
            "mid": [],
            "junior": []
        }

        # contains all risk levels of the selected stories
        blockers_risk_all = []
        fitness_vals = []

        #         sel_items_indices = [i for i in range(len(class_element.chosen_items)) if class_element.chosen_items[i] == 1]
        sel_items_indices = [i for i in range(len(class_element.chosen_items)) if class_element.chosen_items[i] == 1]

        # for each story index
        for i in sel_items_indices:

            #             list(stories_dict.values())[0]

            est_times["all"].append(list(stories.values())[i]["est"])

            if list(stories.values())[i]["compl_lvl"] == 0:
                est_times["low_compl"].append(list(stories.values())[i]["est"])

            elif list(stories.values())[i]["compl_lvl"] == 1:
                est_times["medium_compl"].append(list(stories.values())[i]["est"])

            elif list(stories.values())[i]["compl_lvl"] == 2:
                est_times["high_compl"].append(list(stories.values())[i]["est"])

            else:
                raise Exception("Condition for complexity level is not met.")

            # Append all risk values
            blockers_risk_all.append(list(stories.values())[i]["block_risk"])

            # fitness values (bv*mutual_prio)

            fitness_vals.append(
                list(stories.values())[i]["bv"] * list(stories.values())[i]["mutual_prio"]
            )

        avg_risk = sum(blockers_risk_all) / len(blockers_risk_all)

        # calculate total capacity of developers
        tot_capacity["senior"].append(dev_pool["Senior"] * pi_time_limit)
        tot_capacity["mid"].append(dev_pool["Mid"] * pi_time_limit)
        tot_capacity["junior"].append(dev_pool["Junior"] * pi_time_limit)

        # line 100
        # Calculate the actual fitness

        class_element.fitness = sum(fitness_vals)

        # Check if any of the restrictions are valid:

        total_team_capacity = pi_time_limit * sum(dev_pool.values())

        # 1. Total estimation time of all selected stories should not exceed the total PI working days.
        if sum(est_times["all"]) > total_team_capacity:
            class_element.fitness = 0
            # print("Total est time exceeds the total PI working days; Fitness = 0")
            # print("Total est time:", sum(est_times["all"]))
            # print("Total team working days:", total_team_capacity)
            # print("Total selected stories: ", len(est_times["all"]))
            continue

        # 2. Level of complexity should correspond to the seniority of the devs
        if sum(est_times["high_compl"]) > sum(tot_capacity["senior"]):
            # print(
            #     "There's not enough senior devs to handle all tasks with high complexity; Fitness = 0")  # print(f"Total est time of High complexity tasks: {sum(est_times["high_compl"])}")
            # print("Total est time of High complexity tasks:",
            #       sum(est_times['high_compl']))  # print(f"Total capacity of Senior devs: {sum(tot_capacity["senior"]}")
            # print("Total capacity of Senior devs: ", sum(tot_capacity["senior"]))
            class_element.fitness = 0
            continue

        if sum(est_times["medium_compl"]) > sum(tot_capacity["mid"]):
            class_element.fitness = 0
            # print("There's not enough mid devs to handle all tasks with medium complexity; Fitness = 0")
            #             print(f"Total est time (Medium complexity tasks): {sum(est_times["medium_compl"])}")
            # print("Total est time (Medium complexity tasks): ", sum(est_times["medium_compl"]))
            #             print(f"Total capacity of Senior devs: {sum(tot_capacity["mid"]}")
            # print("Total capacity of Mid devs: ", sum(tot_capacity["mid"]))
            continue

        if sum(est_times["low_compl"]) > sum(tot_capacity["junior"]):
            class_element.fitness = 0

            # print("There's not enough junior devs to handle all tasks with low complexity; Fitness = 0")
            #             print(f"Total est time (Low complexity tasks): {sum(est_times["low_compl"])}")
            # print("Total est time (Low complexity tasks): ", sum(est_times["low_compl"]))
            #             print(f"Total capacity of Senior devs: {sum(tot_capacity["junior"]}")
            # print("Total capacity of Junior devs: ", sum(tot_capacity["junior"]))
            continue

            # 4. If average risk for blockers is higher than 1, the total estimation of the stories would change with a % according to the avg value.

        if avg_risk > 1:

            # print("Avg risk is bigger than 1. Adjusting the total estimation of all selected stories.")

            tot_est_time = sum(est_times["all"]) * avg_risk
            # print("Total adj estimated time: ", tot_est_time)
            # print("total team capacity: ", total_team_capacity)
            if tot_est_time > total_team_capacity:
                class_element.fitness = 0

                # print("Total adjusted estimated time exceeds the total working days; Fitness = 0")
                # print(f"Total adj estimated time: {tot_est_time}")
                #                 print("Total adj estimated time: ", tot_est_time)
                #                 print(f"Total working days: {pi_time_limit}")
                # print("total team capacity: ", total_team_capacity)
                continue

            # else:
            #     class_element.fitness = sum(fitness_vals)

        # 5. 80% of the total capacity of the devs should be used (per Seniority level)  - check: Estimated time / capacity < 80%

        senior_cap_used_ratio = sum(est_times["high_compl"]) / sum(tot_capacity["senior"])
        mid_cap_used_ratio = sum(est_times["medium_compl"]) / sum(tot_capacity["mid"])

        if senior_cap_used_ratio < 0.8:
            class_element.fitness = senior_cap_used_ratio * sum(fitness_vals)

            # print("80% or lower capacity of the Senior devs has been used; Fitness = ", class_element.fitness)
            # print("Sum of high compl estimated time: ", sum(est_times["high_compl"]))
            # print("Sum of total capacity of Senior devs: ", sum(tot_capacity["senior"]))
            # print("% of capacity used: ", sum(est_times["high_compl"]) / sum(tot_capacity["senior"]))

            #             print(f"{np.round(sum(est_times["high_compl"]) / sum(tot_capacity["senior"]))}  of capacity has been used")
            continue

        if mid_cap_used_ratio < 0.8:
            class_element.fitness = mid_cap_used_ratio * sum(fitness_vals)
            # print("80% or lower capacity of the Mid devs has been used; Fitness = ", class_element.fitness)
            #             print(f"{np.round(sum(est_times["medium_compl"]) / sum(tot_capacity["mid"]))}  of capacity has been used")
            continue

        else:
            class_element.fitness = sum(fitness_vals)

        print(f"Fitness score for class element number {iteration}: {class_element.fitness}")

    return class_elements

    #         scores = []
    #         weights = []
    #         sel_items_indices = [i for i in range(len(member.chosen_belongings)) if member.chosen_belongings[i] == 1]

    #         for i in sel_items_indices:
    #             scores.append(list(member.items_dict.values())[i][1])
    #             weights.append(list(member.items_dict.values())[i][0])

    #         member.fitness = sum(scores) if sum(weights) < max_thresh else 0

    #     return members


class Selection:

    @staticmethod
    def top_elements_selection(class_elements, thresh):
        class_elements = sorted(class_elements, key=lambda class_element: class_element.fitness, reverse=True)
        # print('\n'.join(map(str, members)))
        class_elements = class_elements[:int(thresh * len(class_elements))]

        # print("Selection DONE! ", "Selected n of members: ", len(members))
        return class_elements

    @staticmethod
    def tournament_selection(class_elements, thresh):

        # print("ENTERING TOURNAMENT SELECTION..")
        # if members = 13, thresh=0.3 ===== n_tournamnets = 3,9 (4; k_parts = 3.25(3),
        winners = []

        n_tournaments = int(thresh * len(class_elements))
        random.shuffle(class_elements)
        k_parts = len(class_elements) // n_tournaments

        # print("N_tournaments: ", n_tournaments)
        for i in range(n_tournaments):

            from_slice = k_parts*i
            to_slice = from_slice + k_parts if i != n_tournaments else None
            # if i != n_tournaments - 1:
            #     to_slice = from_slice + k_parts
            # else:
            #     to_slice = None

            tourn_elements = class_elements[from_slice:to_slice]
            # print("Tournament members: ")
            # print("\n".join(map(str, tourn_elements)))
            winner = sorted(tourn_elements, key=lambda tourn_member: tourn_member.fitness, reverse=True)[0]

            # print(f"Winner: {''.join(winner.string)}, Fitness score: {winner.fitness}")
            winners.append(winner)

        # print("Selection DONE! ", "Selected n of winners: ", len(winners))
        # print()
        return winners

    @staticmethod
    def roulette_selection(class_elements, thresh):

        n_winners = int(thresh * len(class_elements))
        elements_weights = [class_element.fitness for class_element in class_elements]

        roulette_choices = random.choices(class_elements, weights=elements_weights, k=n_winners)
        # winners.append(roulette_choices)

        return roulette_choices


class Evolution:

    @staticmethod
    def crossover(class_elements, rate):

        # print("ENTERING CROSSOVER...")

        offspring = []

        i = 0
        while i < ((population_size - len(class_elements)) // 2):
            parent1, parent2 = random.choices(class_elements, k=2)
            if parent1 != parent2:

                slice = int(rate * len(parent1.chosen_items))

                child1 = Initialization(size=Initialization.n_stories)
                child2 = Initialization(size=Initialization.n_stories)

                child1.chosen_items = parent1.chosen_items[:slice] + parent2.chosen_items[slice:]
                child2.chosen_items = parent2.chosen_items[:slice] + parent1.chosen_items[slice:]

                #TODO could make the split(rate) random
                offspring.append(child1)
                offspring.append(child2)

                i += 1


        class_elements.extend(offspring)
        # print("Crossover DONE!", "Len of members after crossover: ", len(members))
        return class_elements



    @staticmethod
    def mutation(class_elements, rate):
        # print("=== Mutation Starting ...")
        for class_element in class_elements:

            for idx, param in enumerate(class_element.chosen_items):

                # print("idx: ", idx, " param: ", param)
                if rate > random.uniform(0, 1):

                    # print(f"...Mutating letter '{param}' with index '{idx}' in string {member.string}")
                    class_element.chosen_items = class_element.chosen_items[:idx] + [random.choice([0, 1])] + class_element.chosen_items[idx + 1:]

        # print("=== Mutation DONE!")
        return class_elements




# 5. Run the stupid script

ga()
