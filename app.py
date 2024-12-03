from flask import Flask, render_template, request
import pandas as pd
import random
from flask import jsonify


class Dish:
    def __init__(self, id, name, calories, protein):
        self.id = id
        self.name = name
        self.calories = calories
        self.protein = protein



# %%
recipes = pd.read_csv('data/recipes.csv')
item_list = []
for index, row in recipes.iterrows():
    id, n, c, p = row['id'], row['name'], row['calories'], row['protein']
    new_dish = Dish(int(id), str(n), float(c), float(p))
    item_list.append(new_dish)


# %%
def create_random_solution(i_list, dish_num):
    n = len(i_list)
    indices = list(range(n))
    random.shuffle(indices)
    selected_indices = sorted(indices[:dish_num])

    solution = [0] * n
    for index in selected_indices:
        solution[index] = 1
    return solution


# %%
def calculate_calories(i_list, s_list):
    total_calories = 0
    total_protein = 0
    for i in range(0, len(s_list)):
        if s_list[i] == 1:
            total_calories += i_list[i].calories
            total_protein += i_list[i].protein

    return total_calories


# %%
def calculate_protein(i_list, s_list):
    total_calories = 0
    total_protein = 0
    for i in range(0, len(s_list)):
        if s_list[i] == 1:
            total_calories += i_list[i].calories
            total_protein += i_list[i].protein
    return total_protein


# %%
def get_dishes(i_list, s_list):
    dishes = []
    for i in range(len(s_list)):
        if s_list[i] == 1:
            dish = {
                "name": i_list[i].name,
                "calories": i_list[i].calories,
                "protein": i_list[i].protein,
                "id": i_list[i].id
            }
            dishes.append(dish)
    return dishes


# %%
def fitness(i_list, s_list, limit1, limit2):
    total_calories = 0
    total_protein = 0
    for i in range(0, len(s_list)):
        if s_list[i] == 1:
            total_calories += i_list[i].calories
            total_protein += i_list[i].protein
            if total_calories > limit1 or total_protein > limit2:
                return 0
    return total_calories * 0.6 + total_protein * 0.4


# %%
def initial_population(pop_size, i_list, dish_num):
    population = []
    i = 0
    while i < pop_size:
        new_solution = create_random_solution(i_list, dish_num)

        population.append(new_solution)
        i += 1
    return population


# %%
def tournament_selection(pop, limit1, limit2):
    ticket_1 = random.randint(0, len(pop) - 1)
    ticket_2 = random.randint(0, len(pop) - 1)
    if fitness(item_list, pop[ticket_1], limit1, limit2) > fitness(item_list, pop[ticket_2], limit1, limit2):
        winner = pop[ticket_1]
    else:
        winner = pop[ticket_2]

    return winner


# %%
def crossover(p_1, p_2, dish_num):

    child_1 = [0] * len(p_1)
    child_2 = [0] * len(p_1)
    ones_count_1 = 0
    ones_count_2 = 0


    for i in range(len(p_1)):
        if random.random() < 0.5:
            child_1[i] = p_1[i]
            child_2[i] = p_2[i]
        else:
            child_1[i] = p_2[i]
            child_2[i] = p_1[i]

        ones_count_1 += child_1[i]
        ones_count_2 += child_2[i]


    while ones_count_1 > dish_num:
        idx = random.choice([j for j in range(len(child_1)) if child_1[j] == 1])
        child_1[idx] = 0
        ones_count_1 -= 1
    while ones_count_1 < dish_num:
        idx = random.choice([j for j in range(len(child_1)) if child_1[j] == 0])
        child_1[idx] = 1
        ones_count_1 += 1


    while ones_count_2 > dish_num:
        idx = random.choice([j for j in range(len(child_2)) if child_2[j] == 1])
        child_2[idx] = 0
        ones_count_2 -= 1
    while ones_count_2 < dish_num:
        idx = random.choice([j for j in range(len(child_2)) if child_2[j] == 0])
        child_2[idx] = 1
        ones_count_2 += 1

    return child_1, child_2


# %%
def mutation(chromosome):
    # Find all indices where the chromosome has a '1' and a '0'
    ones_indices = [i for i in range(len(chromosome)) if chromosome[i] == 1]
    zeros_indices = [i for i in range(len(chromosome)) if chromosome[i] == 0]

    # If not enough '1's or '0's to swap, just return the chromosome as is
    if not ones_indices or not zeros_indices:
        return chromosome

    # Randomly select one index from ones_indices and one from zeros_indices
    mutation_index_1 = random.choice(ones_indices)
    mutation_index_2 = random.choice(zeros_indices)

    # Perform the swap
    new_chromosome = chromosome[:]
    new_chromosome[mutation_index_1], new_chromosome[mutation_index_2] = new_chromosome[mutation_index_2], \
        new_chromosome[mutation_index_1]

    return new_chromosome


# %%
def create_generation(pop, mut_rate, limit1 , limit2  ,dish_num):
    new_gen = []
    for i in range(0, int(len(pop)*0.5)):
        parent_1 = tournament_selection(pop , limit1 , limit2)
        parent_2 = tournament_selection(pop , limit1 , limit2)
        child1,child2 = crossover(parent_1, parent_2, dish_num)

        if random.random() < mut_rate:
            child1 = mutation(child1)

        new_gen.append(child1)
        new_gen.append(child2)
    return new_gen


# %%
def best_solution(generation, i_list, limit1, limit2):
    genome = generation[0]
    best = -5
    best1 = 0
    best2 = 0
    for i in range(0, len(generation)):
        temp = fitness(i_list, generation[i], limit1, limit2)
        if temp > best:
            best = temp
            best1 = calculate_calories(i_list, generation[i])
            best2 = calculate_protein(i_list, generation[i])
            genome = generation[i]
    return best, genome, best1, best2


# %%
def genetic_algorithm(p_size, gen_size, mutation_rate, i_list, limit1, limit2, dish_num):
    value_list = []
    pop = initial_population(p_size, i_list, dish_num)
    for i in range(0, gen_size):
        pop = create_generation(pop, mutation_rate, limit1, limit2, dish_num)
        value_list.append(best_solution(pop, i_list, limit1, limit2))
    return pop, value_list


def meal_planner(pop_size, g_size, mut_rate, item_list, test1, test2, dish_num):
    latest_pop1, v_list1 = genetic_algorithm(
        p_size=pop_size,
        gen_size=g_size,
        mutation_rate=mut_rate,
        i_list=item_list, limit1=test1, limit2=test2, dish_num=dish_num)
    if max(v_list1, key=lambda item: item[0])[0] != 0:
        df = pd.DataFrame(data=get_dishes(item_list, max(v_list1, key=lambda item: item[0])[1]))
        return df
    else:
        return None


app = Flask(__name__)


from flask import jsonify, render_template

from flask import jsonify, render_template, request

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        meal_count = int(request.json['mealCount'])  # Expect JSON body
        calories = float(request.json['calories'])
        protein = float(request.json['protein'])

        df = meal_planner(pop_size=100, g_size=100, mut_rate=0.01, item_list=item_list,
                          test1=calories, test2=protein, dish_num=meal_count * 2)

        if df is not None:
            total_calories = sum(df.to_dict()['calories'].values())
            total_calories = round(total_calories, 2)
            total_protein = sum(df.to_dict()['protein'].values())
            total_protein = round(total_protein, 2)

            # Round the calories and protein values in the dataframe
            for i in df.to_dict()['calories'].values():
                round(i, 2)

            for i in df.to_dict()['protein'].values():
                round(i, 2)

            # Split the recipes into meals (groups of 2 recipes)
            meals = df.to_dict('records')
            meal_list = []

            for i in range(0, len(meals), 2):
                meal = meals[i:i + 2]
                meal_total_calories = sum(m['calories'] for m in meal)
                meal_total_protein = sum(m['protein'] for m in meal)

                meal_list.append({
                    'recipes': meal,
                    'total_calories': round(meal_total_calories, 2),
                    'total_protein': round(meal_total_protein, 2)
                })

            # Return JSON response with meals and calculated totals
            return jsonify({
                'meals': meal_list,
                'total_calories': total_calories,
                'total_protein': total_protein
            })
        else:
            return jsonify({
                'error': "No meals found for the given preferences."
            }), 400








if __name__ == '__main__':
    app.run()
