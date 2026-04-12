import torch
import torch.nn as nn
import torchsde
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import copy
import warnings

# --- 1. SDE MODEL DEFINITION ---
class NeuralSDE(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.input_dim = 3
        # Modificado: La entrada ahora es 4 (Fuel, Vel, Alt + Clima/Tiempo)
        self.f_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 3)
        )
        self.g_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
            nn.Softplus()
        )
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        w = torch.zeros((y.size(0), 1), device=y.device)
        if hasattr(self, 'weather_fn'):
            w = self.weather_fn(t)
        yw = torch.cat([y, w], dim=-1)
        return self.f_net(yw)

    def g(self, t, y):
        w = torch.zeros((y.size(0), 1), device=y.device)
        if hasattr(self, 'weather_fn'):
            w = self.weather_fn(t)
        yw = torch.cat([y, w], dim=-1)
        return self.g_net(yw)

def get_model(device):
    model = NeuralSDE().to(device)
    # We initialize with a bias that ensures fuel (idx 0) inherently depletes over time
    # to realistically map flight time to fuel consumption in absence of a loaded model file.
    with torch.no_grad():
        model.f_net[-1].bias[0] = -0.5  # Drift of fuel is strongly negative
    return model

# --- 2. ENVIRONMENT / WEATHER MODEL ---
def weather_penalty(x, y):
    """
    Returns a penalty multiplier in [0, 1] representing severe weather/storms (e.g. strong headwinds).
    Storm 1 is at (40, 60) and Storm 2 is at (70, 30).
    """
    storm1 = np.exp(-((x - 40)**2 + (y - 50)**2) / 200.0)
    storm2 = np.exp(-((x - 70)**2 + (y - 30)**2) / 200.0)
    return 0.8 * storm1 + 0.6 * storm2

# --- 3. B-SPLINE AND ROUTE GENERATION ---
def generate_route(control_points, start_pt, end_pt, num_samples=100):
    """
    control_points: array of shape (K, 2)
    Returns x, y interpolated paths creating a very smooth B-spline.
    """
    pts = np.vstack([start_pt, control_points, end_pt])
    K_total = len(pts)
    t = np.linspace(0, 1, K_total)
    
    # B-spline interpolation ensures aerodynamic smoothness avoiding zig-zag
    spline = make_interp_spline(t, pts, k=min(3, K_total - 1))
    
    t_samples = np.linspace(0, 1, num_samples)
    route = spline(t_samples)
    return route[:, 0], route[:, 1]

# --- 4. EVALUATION / FITNESS (NEURALSDE) ---
def evaluate_population(population, start_pt, end_pt, model, y0, device):
    fitnesses = []
    y0_tensor = torch.tensor(y0, dtype=torch.float32, device=device).unsqueeze(0)
    
    v_base = 50.0 # Standard cruise speed
    routes_xy = []
    
    for chromo in population:
        # Generate the continuous spline
        rx, ry = generate_route(chromo, start_pt, end_pt, num_samples=100)
        routes_xy.append((rx, ry))
        
        # Path distance 
        dx = np.diff(rx)
        dy = np.diff(ry)
        dists = np.sqrt(dx**2 + dy**2)
        
        # Weather interaction
        mid_x = (rx[:-1] + rx[1:]) / 2
        mid_y = (ry[:-1] + ry[1:]) / 2
        penalties = weather_penalty(mid_x, mid_y)
        
        # Effective velocity drops as penalty (storms) increase
        v_eff = np.maximum(v_base * (1.0 - penalties), 10.0) 
        
        # Total time taken to execute the route
        time_taken = np.sum(dists / v_eff)
        t_max = float(time_taken / 2.0)
        
        # Forward pass through NeuralSDE to get fuel burn
        t_span = torch.linspace(0, t_max, 10).to(device)
        
        # --- NUEVO: Inyectar datos del Clima dinámicamente al Modelo SDE ---
        penalties_tensor = torch.tensor(penalties, dtype=torch.float32, device=device)
        def dynamic_weather(t):
            # Mapea un punto en el tiempo t a la penalización meteorológica espacial
            ratio = t.item() / t_max if t_max > 0 else 0
            idx = int(ratio * (len(penalties_tensor) - 1))
            idx = max(0, min(idx, len(penalties_tensor) - 1))
            # Devuelve [batch=1, feat=1] para hacer torch.cat adecuadamente
            return penalties_tensor[idx].unsqueeze(0).unsqueeze(0)
            
        model.weather_fn = dynamic_weather
        
        with torch.no_grad():
            traj = torchsde.sdeint(model, y0_tensor, t_span, method='euler', dt=0.05)
            fuel_remaining = traj[-1, 0, 0].item()
            
        if np.isnan(fuel_remaining):
            fuel_remaining = -np.inf
            
        fitnesses.append(fuel_remaining)
        
    return np.array(fitnesses), routes_xy

# --- 5. GENETIC ALGORITHM ---
def ga_optimize(start_pt, end_pt, model, y0, device, pop_size=50, generations=30, K=6, ngen=25):
    print("\n[GA] Initializing population...")
    population = []
    sx, sy = start_pt
    ex, ey = end_pt
    # Heuristic: base control points line up on the direct ray between A and B
    straight_line_x = np.linspace(sx, ex, K+2)[1:-1]
    straight_line_y = np.linspace(sy, ey, K+2)[1:-1]
    base_pts = np.column_stack([straight_line_x, straight_line_y])
    
    vec = end_pt - start_pt
    # Introduce random divergence to widely explore the 100x100 space
    for _ in range(pop_size):
        noise = np.random.normal(scale=25.0, size=base_pts.shape)
        pt = base_pts + noise
        # Sort points by their projection towards the destination to prevent knots/loops
        pt = pt[np.argsort(np.dot(pt - start_pt, vec))]
        population.append(pt)
        
    best_fitness_history = []
    best_route_xy = None
    
    # Evaluate Initial Population
    fitnesses, routes_xy = evaluate_population(population, start_pt, end_pt, model, y0, device)
    
    current_record_fitness = -np.inf
    generations_without_improvement = 0
    
    for gen in range(generations):
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)
        
        if best_fitness > current_record_fitness + 1e-4:
            current_record_fitness = best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        if gen == generations - 1:
            best_route_xy = routes_xy[best_idx]
            
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"[GA] Generation {gen+1:02d}/{generations} | Best Fuel Remaining: {best_fitness:.4f}")
            
        if generations_without_improvement >= ngen:
            print(f"[GA] Early stopping triggered at generation {gen+1}. No improvement for {ngen} generations.")
            best_route_xy = routes_xy[best_idx]
            break
        
        # Elitism & Memory
        new_population = []
        new_routes_xy = []
        new_fitnesses = []
        
        top_indices = np.argsort(fitnesses)[-4:]  # Elitism: retain top 4
        for idx in top_indices:
            new_population.append(copy.deepcopy(population[idx]))
            new_routes_xy.append(routes_xy[idx])
            new_fitnesses.append(fitnesses[idx])
            
        # Generate offspring
        offspring_to_generate = pop_size - len(top_indices)
        offspring_pop = []
        
        for _ in range(offspring_to_generate):
            # Tournament selection
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            p1 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
            
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            p2 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
            
            # Crossover
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            
            # High-Variance Mutation (Explore better):
            # Evaluate independent mutation for EVERY control point
            for mut_idx in range(K):
                if np.random.rand() < 0.35:
                    child[mut_idx] += np.random.normal(scale=10.0, size=2)
            
            # Eliminate knots/loops dynamically by re-sorting the control points sequentially
            child = child[np.argsort(np.dot(child - start_pt, vec))]
                
            offspring_pop.append(child)
            
        # Evaluate only the new offspring
        offspring_fitnesses, offspring_routes = evaluate_population(offspring_pop, start_pt, end_pt, model, y0, device)
        
        # Combine elites with offspring
        population = new_population + offspring_pop
        routes_xy = new_routes_xy + offspring_routes
        fitnesses = np.array(new_fitnesses + list(offspring_fitnesses))
        
    print(f"[GA] Final Best Fuel Remaining: {best_fitness_history[-1]:.4f}\n")
    return best_route_xy, best_fitness_history

# --- 6. PLOTTING & MAIN EXPORT ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = get_model(device)
    y0 = [1.0, 0.8, 0.5] # Fuel, Velocity, Altitude
    
    start_pt = np.array([5.0, 5.0])
    end_pt = np.array([95.0, 95.0])
    
    best_route, fitness_history = ga_optimize(start_pt, end_pt, model, y0, device, pop_size=50, generations=400, K=5, ngen=50)
    
    # ---------------- Plotting Background and Result ---------------- #
    plt.figure(figsize=(10, 8))
    
    # Contour weather map
    X, Y = np.meshgrid(np.linspace(0, 100, 200), np.linspace(0, 100, 200))
    Z = weather_penalty(X, Y)
    contour = plt.contourf(X, Y, Z, levels=30, cmap='Reds', alpha=0.6)
    plt.colorbar(contour, label='Weather Severity (Storms)')
    
    # Nodes and Base Route
    plt.scatter([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color='blue', s=150, zorder=5, label='Airports')
    plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 'b--', alpha=0.5, linewidth=2, label='Direct Route')
    
    # Best Interpolated B-Spline
    if best_route is not None:
        plt.plot(best_route[0], best_route[1], 'g-', linewidth=4, zorder=4, label='Optimized B-Spline Route')
        
    plt.title('B-Spline Genetic Algorithm Route Optimization', fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('route_optimization_result.png', bbox_inches='tight')
    plt.show()
    
    # Convergence Plot
    plt.figure(figsize=(8, 4))
    plt.plot(fitness_history, 'b-o', linewidth=2)
    plt.title('GA Convergence', fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Fuel Remaining (Fitness)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('fitness_convergence.png', bbox_inches='tight')
    plt.show()
