import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deap import base, creator, tools, algorithms

# --- CONFIGURATION ---
PALLET_W, PALLET_H = 100, 100
NUM_ITEMS = 18
POP_SIZE = 150
GENERATIONS = 150
P_CROSSOVER = 0.6
P_MUTATION = 0.3
TOURNAMENT_SIZE = 5

random.seed(42)
ITEMS_DATA = [(random.randint(10, 25), random.randint(10, 25)) for _ in range(NUM_ITEMS)]

# --- CORE LOGIC ---
def evaluate(individual):
    total_area = 0
    overlap_area = 0
    out_of_bounds = 0
    dist_from_center_penalty = 0
    
    placed_rects = []
    pallet_cx, pallet_cy = PALLET_W / 2, PALLET_H / 2

    for gene in individual:
        item_id, rot, x, y = gene
        orig_w, orig_h = ITEMS_DATA[item_id]
        # Rotate if gene says so (0 or 1)
        w, h = (orig_h, orig_w) if rot == 1 else (orig_w, orig_h)
        
        # Calculate item center
        item_cx, item_cy = x + w/2, y + h/2
        
        # 1. Boundary Check
        if x < 0 or y < 0 or x + w > PALLET_W or y + h > PALLET_H:
            out_of_bounds += (w * h)
        else:
            # 2. Collision Check
            current_overlap = 0
            for (rx, ry, rw, rh) in placed_rects:
                dx = min(x + w, rx + rw) - max(x, rx)
                dy = min(y + h, ry + rh) - max(y, ry)
                if dx > 0 and dy > 0:
                    current_overlap += dx * dy
            
            overlap_area += current_overlap
            placed_rects.append((x, y, w, h))
            total_area += w * h
            
            # 3. Stability (Distance to Center)
            dist_from_center_penalty += (abs(item_cx - pallet_cx) + abs(item_cy - pallet_cy))

    score = (total_area * 10) - (overlap_area * 100) - (out_of_bounds * 100) - (dist_from_center_penalty * 0.5)
    return (score,)

def custom_mutate(individual, indpb):
    """Hybrid mutation: Rotate, Shift, or Swap."""
    for i in range(len(individual)):
        if random.random() < indpb:
            r = random.random()
            if r < 0.3:
                # Mutation A: Rotate
                individual[i][1] = 1 - individual[i][1]
            elif r < 0.9:
                # Mutation B: Shift (Nudging)
                shift = random.randint(-10, 10)
                if random.random() < 0.5:
                    individual[i][2] = max(0, min(PALLET_W, individual[i][2] + shift))
                else:
                    individual[i][3] = max(0, min(PALLET_H, individual[i][3] + shift))
            else:
                # Mutation C: Swap Order
                idx2 = random.randint(0, len(individual)-1)
                individual[i], individual[idx2] = individual[idx2], individual[i]
    return individual,

# --- GA SETUP ---
try:
    del creator.FitnessMax
    del creator.Individual
except Exception:
    pass

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 lambda: [[i, random.randint(0, 1), random.randint(0, PALLET_W-10), random.randint(0, PALLET_H-10)] for i in range(NUM_ITEMS)])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# --- VISUALIZATION ---
def draw_solution(individual, filename="result_final.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, PALLET_W)
    ax.set_ylim(0, PALLET_H)
    
    total_item_area = 0
    moment_x = 0
    moment_y = 0
    
    # Draw Pallet Border
    ax.add_patch(patches.Rectangle((0, 0), PALLET_W, PALLET_H, fill=False, edgecolor='black', linewidth=3))
    
    for gene in individual:
        item_id, rot, x, y = gene
        w, h = ITEMS_DATA[item_id]
        if rot == 1: w, h = h, w
        
        # Physics calculations
        area = w * h
        total_item_area += area
        moment_x += (x + w/2) * area
        moment_y += (y + h/2) * area
        
        # Draw Item
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=np.random.rand(3,), alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, f"{item_id}", ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # Calculate Center of Mass
    if total_item_area > 0:
        com_x = moment_x / total_item_area
        com_y = moment_y / total_item_area
    else:
        com_x, com_y = PALLET_W/2, PALLET_H/2

    pallet_area = PALLET_W * PALLET_H
    fill_rate = (total_item_area / pallet_area) * 100
    
    # --- STABILITY VIZ ---

    ax.plot(PALLET_W/2, PALLET_H/2, 'rx', markersize=12, markeredgewidth=2, label='Pallet Center')
    ax.plot(com_x, com_y, 'bo', markersize=10, label='Load CoG')
    ax.plot([PALLET_W/2, com_x], [PALLET_H/2, com_y], 'k--', linewidth=1, alpha=0.5)

    ax.set_title(f"Packing Optimization\nFill Rate: {fill_rate:.2f}% | Fitness: {individual.fitness.values[0]}")
    ax.legend(loc='upper right')
    
    plt.savefig(filename, dpi=120)
    print(f"\n[INFO] Visualization saved to: {filename}")
    print(f"[INFO] Final Fill Rate: {fill_rate:.2f}%")

# --- MAIN EXECUTION ---
def main():
    pop = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    random.seed(5)
    print(f"--- STARTING EVOLUTION ---")
    print(f"Items: {NUM_ITEMS} | Generations: {GENERATIONS} | Population: {POP_SIZE}")
    print("Please wait, calculating...")
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=GENERATIONS, stats=stats, verbose=True)
    
    best_ind = tools.selBest(pop, 1)[0]
    
    print("\n--- FINAL STATISTICS ---")
    print(f"Best Fitness Found: {best_ind.fitness.values[0]:.2f}")
    
    draw_solution(best_ind)

if __name__ == "__main__":
    main()