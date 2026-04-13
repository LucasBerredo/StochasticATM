import torch
import torch.nn as nn
import torchsde
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import copy
import warnings

# --- 1. SDE MODEL DEFINITION ---
class NeuralFuelSDE(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        # Red recibe 6 features: [Fuel] + Controles [vel, alt, vrate, track_sin, track_cos]
        self.f_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)  # Predice solo derivada de 1 Feature: Fuel
        )
        self.g_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def f(self, t, y):
        # Fetch external path controls for time t: [Vel, Alt, Vrate, Sin, Cos]
        controls = torch.zeros((y.size(0), 5), device=y.device)
        if hasattr(self, 'control_fn'):
            c_val = self.control_fn(t)
            if c_val.size(0) == 1 and y.size(0) > 1:
                c_val = c_val.expand(y.size(0), -1)
            controls = c_val
            
        yw = torch.cat([y, controls], dim=-1)
        res = self.f_net(yw)
        # Physical constraint: Fuel (idx 0) can only decrease
        fuel_gate = -torch.nn.functional.softplus(res)
        return fuel_gate

    def g(self, t, y):
        controls = torch.zeros((y.size(0), 5), device=y.device)
        if hasattr(self, 'control_fn'):
            c_val = self.control_fn(t)
            if c_val.size(0) == 1 and y.size(0) > 1:
                c_val = c_val.expand(y.size(0), -1)
            controls = c_val
            
        yw = torch.cat([y, controls], dim=-1)
        return self.g_net(yw)

def get_model(device):
    model = NeuralFuelSDE().to(device)
    return model

# --- 2. TRAINING PIPELINE ---
def train_step(model, batch_y0, batch_t, batch_y_true_fuel, optimizer, is_lbfgs=False):
    model.train()
    def closure():
        optimizer.zero_grad()    
        pred_y = torchsde.sdeint_adjoint(model, batch_y0, batch_t, dt=0.01, method='euler')
        loss = nn.MSELoss()(pred_y, batch_y_true_fuel)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return loss
    
    if is_lbfgs:
        loss = optimizer.step(closure)
    else:
        loss = closure()
        optimizer.step()
    return loss.item()

def train_exogenous_model(model, path, device, epochs=40, lbfgs_start_epoch=30):
    print(f"\n[Train] Loading historical data from {path}...")
    data = torch.load(path, map_location=device, weights_only=False)
    t = data['t'].to(device)
    y_true_raw = data['y_true'].to(device)
    
    # Sanitización de datos: Eliminar baches corruptos del simulador (NaNs)
    valid_mask = ~torch.isnan(y_true_raw).any(dim=0).any(dim=-1)
    y_true = y_true_raw[:, valid_mask, :]
    print(f"[Train] Sanitized Dataset: {valid_mask.sum().item()} valid flights kept out of {len(valid_mask)}")
    
    y_true_fuel = y_true[:, :, 0:1] # Solo Fuel [Time, Batch, 1]
    batch_y0 = y_true_fuel[0] # [Batch, 1]
    
    # Controles Dinámicos [Vel, Alt, Vrate, Sin, Cos]
    controls_data = y_true[:, :, 1:6]
    
    t_min, t_max = t.min().item(), t.max().item()
    def training_control_fn(curr_t):
        ratio = (curr_t.item() - t_min) / (t_max - t_min + 1e-6)
        idx = max(0, min(int(ratio * (len(t) - 1)), len(t) - 1))
        return controls_data[idx]
        
    model.control_fn = training_control_fn
    
    # Se añade un learning rate mitigado a petición para asegurar consistencia
    optimizer_adam = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    print("[Train] Starting NeuralFuelSDE Offline Training...")
    for epoch in range(1, epochs + 1):
        if epoch < lbfgs_start_epoch:
            loss = train_step(model, batch_y0, t, y_true_fuel, optimizer_adam)
            opt_name = 'AdamW'
        else:
            loss = train_step(model, batch_y0, t, y_true_fuel, optimizer_lbfgs, True)
            opt_name = 'LBFGS'
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Train] Epoch {epoch:03d} | Loss: {loss:.6f} | Optimizer: {opt_name}")
            
    return model

# --- 3. ENVIRONMENT / WEATHER MODEL ---
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
        
        # --- NUEVO: Inyectar 5 datos dinámicos (Velocity, Alt, Vrate, Track_sin, Track_cos) ---
        bearing = np.arctan2(dy, dx)
        track_sin = np.append(np.sin(bearing), np.sin(bearing)[-1])
        track_cos = np.append(np.cos(bearing), np.cos(bearing)[-1])
        v_eff_pad = np.append(v_eff, v_eff[-1])
        
        v_eff_tensor = torch.tensor(v_eff_pad, dtype=torch.float32, device=device) / 100.0 # Escalado
        alt_tensor = torch.full_like(v_eff_tensor, 0.8) # Altitud constante en ruta
        vrate_tensor = torch.zeros_like(v_eff_tensor) # Elevación plana
        track_sin_tensor = torch.tensor(track_sin, dtype=torch.float32, device=device)
        track_cos_tensor = torch.tensor(track_cos, dtype=torch.float32, device=device)
        
        def flight_control_fn(t):
            # Mapea un punto en el tiempo t a las variables cinemáticas de la ruta
            ratio = t.item() / t_max if t_max > 0 else 0
            idx = int(ratio * (len(v_eff_tensor) - 1))
            idx = max(0, min(idx, len(v_eff_tensor) - 1))
            
            # Devuelve [batch=1, feat=5]
            return torch.stack([v_eff_tensor[idx], alt_tensor[idx], vrate_tensor[idx], track_sin_tensor[idx], track_cos_tensor[idx]]).unsqueeze(0)
            
        model.control_fn = flight_control_fn
        
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
    best_route_control_points = None
    
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
            best_route_control_points = population[best_idx]
            
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"[GA] Generation {gen+1:02d}/{generations} | Best Fuel Remaining: {best_fitness:.4f}")
            
        if generations_without_improvement >= ngen:
            print(f"[GA] Early stopping triggered at generation {gen+1}. No improvement for {ngen} generations.")
            best_route_xy = routes_xy[best_idx]
            best_route_control_points = population[best_idx]
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
    return best_route_xy, best_route_control_points, best_fitness_history

# --- 7. UNCERTAINTY ANALYSIS (SDE FAN CHART) ---
def plot_fuel_fan_chart(trajectories, t_span, ground_truth=None, title="Optimized Route: Fuel Burn Uncertainty"):
    fuel_data = trajectories[:, :, 0].cpu().numpy()
    t = t_span.cpu().numpy()

    median = np.median(fuel_data, axis=1)
    p5 = np.percentile(fuel_data, 5, axis=1)
    p25 = np.percentile(fuel_data, 25, axis=1)
    p75 = np.percentile(fuel_data, 75, axis=1)
    p95 = np.percentile(fuel_data, 95, axis=1)

    plt.figure(figsize=(10, 5), dpi=100)
    plt.fill_between(t, p5, p95, color='royalblue', alpha=0.2, label='90% Confidence')
    plt.fill_between(t, p25, p75, color='royalblue', alpha=0.4, label='50% Confidence')
    plt.plot(t, median, color='darkblue', lw=2, label='Median (Predictive Drift)')

    if ground_truth is not None:
        plt.plot(t, ground_truth, color='red', lw=2, linestyle='--', label='Actual Test Flight (Ground Truth)')

    plt.title(title, fontweight='bold')
    plt.xlabel("Time (Simulated)", fontsize=10)
    plt.ylabel("Fuel Remaining", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.savefig('fuel_analysis_chart.png', bbox_inches='tight')
    plt.show()

def evaluate_test_set(model, path, device):
    print(f"\n[Test] Evaluating model on Unseen Testing Set: {path}...")
    data = torch.load(path, map_location=device, weights_only=False)
    t = data['t'].to(device)
    y_true = data['y_true'].to(device)
    
    # We pick flight #0 from the batch to draw the fan chart
    y_true_fuel = y_true[:, 0:1, 0:1] # [Time, 1, 1]
    y0_single = y_true_fuel[0] # [1, 1]
    
    # Option B: Full kinematics Controls (with backwards compatibility padding for outdated test datasets)
    c_slice = y_true[:, 0:1, 1:6]
    actual_feats = c_slice.shape[-1]
    if actual_feats < 5:
        padding = torch.zeros(len(t), 1, 5 - actual_feats, device=device)
        controls_single = torch.cat([c_slice, padding], dim=-1)
    else:
        controls_single = c_slice
    
    t_min, t_max = t.min().item(), t.max().item()
    def test_control_fn(curr_t):
        ratio = (curr_t.item() - t_min) / (t_max - t_min + 1e-6)
        idx = max(0, min(int(ratio * (len(t) - 1)), len(t) - 1))
        return controls_single[idx]
        
    model.control_fn = test_control_fn
    model.eval()
    
    with torch.no_grad():
        num_sims = 100
        y0_expanded = y0_single.repeat(num_sims, 1)
        trajectories = torchsde.sdeint(model, y0_expanded, t, method='euler', dt=0.01)
        
    ground_truth_np = y_true_fuel[:, 0, 0].cpu().numpy()
    plot_fuel_fan_chart(trajectories, t, ground_truth=ground_truth_np, title="Test Set Evaluation: SDE Prediction vs Ground Truth")

def analyze_route(control_points, start_pt, end_pt, model, y0, device, num_simulations=100):
    print(f"\n[Analysis] Generating {num_simulations} Monte Carlo samples over optimal path...")
    model.eval()
    y0_tensor = torch.tensor(y0, dtype=torch.float32, device=device).unsqueeze(0)
    
    rx, ry = generate_route(control_points, start_pt, end_pt, num_samples=100)
    dx, dy = np.diff(rx), np.diff(ry)
    dists = np.sqrt(dx**2 + dy**2)
    
    mid_x, mid_y = (rx[:-1] + rx[1:]) / 2, (ry[:-1] + ry[1:]) / 2
    penalties = weather_penalty(mid_x, mid_y)
    
    v_base = 50.0
    v_eff = np.maximum(v_base * (1.0 - penalties), 10.0) 
    
    t_max = float(np.sum(dists / v_eff) / 2.0)
    t_span = torch.linspace(0, t_max, 50).to(device)
    
    bearing = np.arctan2(dy, dx)
    track_sin = np.append(np.sin(bearing), np.sin(bearing)[-1])
    track_cos = np.append(np.cos(bearing), np.cos(bearing)[-1])
    v_eff_pad = np.append(v_eff, v_eff[-1])
    
    v_eff_tensor = torch.tensor(v_eff_pad, dtype=torch.float32, device=device) / 100.0
    alt_tensor = torch.full_like(v_eff_tensor, 0.8)
    vrate_tensor = torch.zeros_like(v_eff_tensor)
    track_sin_tensor = torch.tensor(track_sin, dtype=torch.float32, device=device)
    track_cos_tensor = torch.tensor(track_cos, dtype=torch.float32, device=device)
    
    def flight_control_fn(t):
        ratio = t.item() / t_max if t_max > 0 else 0
        idx = max(0, min(int(ratio * (len(v_eff_tensor) - 1)), len(v_eff_tensor) - 1))
        return torch.stack([v_eff_tensor[idx], alt_tensor[idx], vrate_tensor[idx], track_sin_tensor[idx], track_cos_tensor[idx]]).unsqueeze(0)
        
    model.control_fn = flight_control_fn
    
    with torch.no_grad():
        y0_expanded = y0_tensor.repeat(num_simulations, 1)
        trajectories = torchsde.sdeint(model, y0_expanded, t_span, method='euler', dt=0.01)
        
    plot_fuel_fan_chart(trajectories, t_span, ground_truth=None)

# --- 8. PLOTTING & MAIN EXPORT ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = get_model(device)
    
    # ---------------- TRAINING & TESTING PIPELINE ---------------- #
    # Si los archivos de telemetría existen, entrena y testea. Asumimos el subdirectorio especificado.
    try:
        model = train_exogenous_model(model, 'flight_data.pt', device, epochs=100, lbfgs_start_epoch=40)
        evaluate_test_set(model, 'flight_data_test.pt', device)
    except FileNotFoundError:
        print("\n[Warn] flight_data.pt not found. Skipping training -> using untrained structural model.")
    
    # ---------------- GENETIC ALGORITHM OPTIMIZATION ---------------- #
    y0 = [1.0] # Fuel ONLY, shape [1]
    
    start_pt = np.array([5.0, 5.0])
    end_pt = np.array([95.0, 95.0])
    
    best_route, best_control_pts, fitness_history = ga_optimize(start_pt, end_pt, model, y0, device, pop_size=50, generations=400, K=5, ngen=50)
    
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

    # Reproduce Notebook's NeuralSDE analytical behavior
    if best_control_pts is not None:
        analyze_route(best_control_pts, start_pt, end_pt, model, y0, device, num_simulations=100)
