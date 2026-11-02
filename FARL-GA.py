import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import kmeans2
from mpl_toolkits.mplot3d import Axes3D
import csv
from openpyxl import Workbook
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# MARL module: 10 BS as parameter-sharing agents with CTDE architecture
# Serves as enhanced local search within GA, replacing part of coverage_margin_local_search
# Actor: local obs -> displacement (dx, dy); Critic: global state + all actions -> Q-value

class MARLActor(keras.Model):
    def __init__(self, obs_dim, action_dim=2, max_action=30.0):
        super().__init__()
        self.max_action = max_action
        self.fc1 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.mean = layers.Dense(action_dim, kernel_initializer='glorot_uniform')
        self.log_std = layers.Dense(action_dim, kernel_initializer='glorot_uniform')

    def call(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        mean = self.max_action * tf.tanh(self.mean(x))
        log_std = tf.clip_by_value(self.log_std(x), -20.0, 2.0)
        std = tf.exp(log_std)
        return mean, std

    def sample_action(self, obs, training=True):
        obs = tf.cast(obs, tf.float32)
        mean, std = self.call(obs)
        if training:
            noise = tf.random.normal(tf.shape(mean))
            action = mean + std * noise
        else:
            action = mean
        action = tf.clip_by_value(action, -self.max_action, self.max_action)
        log_prob = -0.5 * tf.reduce_sum(((action - mean) / (std + 1e-8))**2, axis=-1) \
                   - tf.reduce_sum(tf.math.log(std + 1e-8), axis=-1) \
                   - 0.5 * tf.cast(tf.shape(action)[-1], tf.float32) * tf.math.log(2.0 * np.pi)
        return action, log_prob


class MARLCritic(keras.Model):
    def __init__(self, global_state_dim, obs_dim, action_dim=2):
        super().__init__()
        input_dim = global_state_dim + obs_dim + action_dim
        self.fc1 = layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.q = layers.Dense(1, kernel_initializer='glorot_uniform')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        q = self.q(x)
        return q


class MARLSearcher:
    """Multi-Agent RL driven local search with CTDE architecture."""
    def __init__(self, n_agents, n, Z, terrain_cost, slope, ux, uy, uz, Cmin,
                 evaluate_fn, deduplicate_fn, max_action=30.0, gamma=0.95,
                 actor_lr=1e-3, critic_lr=2e-3):
        self.n_agents = n_agents
        self.n = n
        self.Z = Z
        self.terrain_cost = terrain_cost
        self.slope = slope
        self.ux, self.uy, self.uz = ux, uy, uz
        self.Cmin = Cmin
        self.evaluate = evaluate_fn
        self.deduplicate_bs = deduplicate_fn
        self.gamma = gamma
        self.max_action = max_action

        self.obs_dim = 9
        self.global_state_dim = 4

        self.actor = MARLActor(self.obs_dim, action_dim=2, max_action=max_action)
        self.critic = MARLCritic(self.global_state_dim, self.obs_dim, action_dim=2)
        self.actor_opt = keras.optimizers.Adam(actor_lr)
        self.critic_opt = keras.optimizers.Adam(critic_lr)

        self.replay_buffer = []
        self.buffer_size = 2000
        self.batch_size = 64

        self.train_rewards = []
        self.train_costs = []
        self.train_coverages = []

        # Initialize weights
        dummy_obs = tf.zeros((1, self.obs_dim))
        _ = self.actor(dummy_obs)
        dummy_input = tf.zeros((1, self.global_state_dim + self.obs_dim + 2))
        _ = self.critic(dummy_input)

    def get_agent_obs(self, sol, agent_id):
        """Local obs: position(2) + terrain(3) + neighbors(2) + local users(2) = 9"""
        bs_x = sol[0::2]
        bs_y = sol[1::2]
        ox, oy = bs_x[agent_id], bs_y[agent_id]
        pos = np.array([ox / self.n, oy / self.n], dtype=np.float32)
        tc = self.terrain_cost[oy - 1, ox - 1] / 200.0
        sl = self.slope[oy - 1, ox - 1] / 90.0
        el = self.Z[oy - 1, ox - 1] / 120.0
        terrain_feat = np.array([tc, sl, el], dtype=np.float32)
        other_dists = np.sqrt((bs_x - ox)**2 + (bs_y - oy)**2)
        other_dists[agent_id] = 1e6
        min_d = other_dists.min() / (self.n * 0.5)
        n_close = (other_dists < 50).sum()
        neighbor_feat = np.array([min_d, n_close / 10.0], dtype=np.float32)
        udx = self.ux - ox
        udy = self.uy - oy
        ud = np.sqrt(udx**2 + udy**2)
        in_range = (ud < 100).sum()
        avg_d = ud.mean() / (self.n * 0.7)
        user_feat = np.array([in_range / 40.0, avg_d], dtype=np.float32)
        return np.concatenate([pos, terrain_feat, neighbor_feat, user_feat])

    def get_global_state(self, it, MaxIt, coverage, cost, diversity=0.5):
        progress = it / MaxIt
        cov_norm = coverage
        cost_norm = min(cost / 5000.0, 2.0)
        return np.array([progress, cov_norm, cost_norm, diversity], dtype=np.float32)

    def step(self, sol, it, MaxIt, current_cov, current_cost):
        sol = self.deduplicate_bs(sol)
        old_cov, _, old_cost = self.evaluate(sol[0::2], sol[1::2])
        global_state = self.get_global_state(it, MaxIt, old_cov, old_cost)

        obs_list = []
        actions = []
        log_probs = []
        for i in range(self.n_agents):
            obs = self.get_agent_obs(sol, i)
            obs_list.append(obs)
            obs_tf = tf.expand_dims(tf.cast(obs, tf.float32), 0)
            action, log_prob = self.actor.sample_action(obs_tf, training=True)
            actions.append(action.numpy()[0])
            log_probs.append(float(log_prob.numpy()[0]))

        new_sol = sol.copy()
        for i in range(self.n_agents):
            dx = int(round(actions[i][0]))
            dy = int(round(actions[i][1]))
            new_sol[2*i] = int(np.clip(new_sol[2*i] + dx, 1, self.n))
            new_sol[2*i + 1] = int(np.clip(new_sol[2*i + 1] + dy, 1, self.n))
        new_sol = self.deduplicate_bs(new_sol)

        new_cov, _, new_cost = self.evaluate(new_sol[0::2], new_sol[1::2])

        # Reward: cost reduction + coverage constraint + overlap penalty + feasibility bonus
        reward_cost = old_cost - new_cost
        reward_cov = -100.0 * max(0, self.Cmin - new_cov)
        bx, by = new_sol[0::2], new_sol[1::2]
        coords = list(zip(bx, by))
        n_unique = len(set(coords))
        reward_overlap = -10.0 * (self.n_agents - n_unique)
        reward_feasible = 5.0 if (new_cov >= self.Cmin and new_cost < old_cost) else 0.0
        total_reward = reward_cost + reward_cov + reward_overlap + reward_feasible

        transition = {
            'obs': np.array(obs_list),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'reward': total_reward,
            'global_state': global_state,
            'old_cost': old_cost,
            'new_cost': new_cost,
            'new_cov': new_cov,
            'new_sol': new_sol.copy()
        }
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.train_rewards.append(total_reward)
        self.train_costs.append(new_cost)
        self.train_coverages.append(new_cov)

        return new_sol, new_cov, new_cost, total_reward

    def train(self):
        """Train Actor (policy gradient with Q as advantage) and Critic (regression on reward)."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        obs_batch = np.concatenate([t['obs'] for t in batch], axis=0)
        action_batch = np.concatenate([t['actions'] for t in batch], axis=0)
        log_prob_batch = np.concatenate([t['log_probs'] for t in batch], axis=0)
        reward_batch = np.array([t['reward'] for t in batch])
        global_state_batch = np.array([t['global_state'] for t in batch])

        q_target = np.tile(reward_batch[:, None], (1, self.n_agents)).reshape(-1, 1)
        global_state_expanded = np.repeat(global_state_batch, self.n_agents, axis=0)
        critic_input = np.concatenate([global_state_expanded, obs_batch, action_batch], axis=1)

        # Train Critic
        with tf.GradientTape() as tape:
            q_values = self.critic(critic_input)
            critic_loss = tf.reduce_mean((q_values - tf.cast(q_target, tf.float32))**2)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Train Actor: maximize log_prob * Q
        with tf.GradientTape() as tape:
            obs_tf = tf.cast(obs_batch, tf.float32)
            mean, std = self.actor(obs_tf)
            noise = tf.random.normal(tf.shape(mean))
            new_actions = mean + std * noise
            new_actions = tf.clip_by_value(new_actions, -self.max_action, self.max_action)
            new_log_prob = -0.5 * tf.reduce_sum(((new_actions - mean) / (std + 1e-8))**2, axis=-1) \
                           - tf.reduce_sum(tf.math.log(std + 1e-8), axis=-1) \
                           - 0.5 * 2.0 * tf.math.log(2.0 * np.pi)
            gs_expanded = tf.cast(global_state_expanded, tf.float32)
            critic_input_new = tf.concat([gs_expanded, obs_tf, new_actions], axis=1)
            q_new = self.critic(critic_input_new)
            q_new = tf.squeeze(q_new, axis=-1)
            actor_loss = -tf.reduce_mean(new_log_prob * tf.stop_gradient(q_new))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return float(critic_loss.numpy()), float(actor_loss.numpy())


def run_simulation(run_id):
    np.random.seed(run_id)

    # Terrain generation
    n = 500
    X, Y = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    Z = np.zeros_like(X, dtype=float)

    num_main = 5
    for _ in range(num_main):
        x0 = np.random.rand() * n
        y0 = np.random.rand() * n
        A = 50 + 40 * np.random.rand()
        sigma_x = 30 + 30 * np.random.rand()
        sigma_y = 80 + 80 * np.random.rand()
        theta = np.random.rand() * np.pi
        Xr = (X - x0) * np.cos(theta) + (Y - y0) * np.sin(theta)
        Yr = -(X - x0) * np.sin(theta) + (Y - y0) * np.cos(theta)
        Z += A * np.exp(-(Xr ** 2 / (2 * sigma_x ** 2) + Yr ** 2 / (2 * sigma_y ** 2)))

    num_hills = 30
    for _ in range(num_hills):
        x0 = np.random.rand() * n
        y0 = np.random.rand() * n
        A = 5 + 20 * np.random.rand()
        sigma = 8 + 25 * np.random.rand()
        Z += A * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

    num_valleys = 8
    for _ in range(num_valleys):
        x0 = np.random.rand() * n
        y0 = np.random.rand() * n
        A = 15 + 30 * np.random.rand()
        sigma = 20 + 50 * np.random.rand()
        Z -= A * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

    Z += 2 * np.random.randn(*Z.shape)
    Z = gaussian_filter(Z, sigma=2)
    Z = Z - Z.min()
    Z = 120 * Z / Z.max()
    Z = Z ** 1.2

    dy, dx = np.gradient(Z)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2)) * 180 / np.pi

    c0 = 100
    lambda1 = 2
    lambda2 = 0.5
    terrain_cost = c0 + lambda1 * slope + lambda2 * Z

    # Users
    num_users = 40
    ux = np.random.randint(1, n + 1, size=num_users)
    uy = np.random.randint(1, n + 1, size=num_users)
    uz = Z[uy - 1, ux - 1]
    ux0 = ux - 1
    uy0 = uy - 1

    # Algorithm parameters
    num_bs = 10
    nVar = num_bs * 2
    VarMin = 1
    VarMax = n

    popSize = 50
    MaxIt = 200
    MAX_TIME = 200
    Cmin = 0.91

    Pt = 46
    noise = -100
    alpha = 0.8

    # Terrain-weighted sampling distribution
    cost_flat = terrain_cost.flatten()
    temp = 0.5
    weights = np.exp(-cost_flat / (temp * np.std(cost_flat) + 1e-6))
    weights /= weights.sum()

    eval_cache = {}

    def deduplicate_bs(sol):
        bs_x = sol[0::2].copy()
        bs_y = sol[1::2].copy()
        seen = set()
        for i in range(num_bs):
            key = (bs_x[i], bs_y[i])
            if key in seen:
                for _ in range(1000):
                    nx = np.random.randint(1, n + 1)
                    ny = np.random.randint(1, n + 1)
                    if (nx, ny) not in seen:
                        bs_x[i], bs_y[i] = nx, ny
                        break
            seen.add((bs_x[i], bs_y[i]))
        out = sol.copy()
        out[0::2] = bs_x
        out[1::2] = bs_y
        return out

    # Cluster-based initialization: k-means on user positions + local min cost
    user_points = np.column_stack([ux.astype(float), uy.astype(float)])
    centroids, _ = kmeans2(user_points, num_bs, seed=run_id, minit='points')
    cluster_sol = np.zeros(nVar, dtype=int)
    r = 25
    for k in range(num_bs):
        cx, cy = int(np.clip(centroids[k, 0], 1, n)), int(np.clip(centroids[k, 1], 1, n))
        x_lo, x_hi = max(1, cx - r), min(n, cx + r)
        y_lo, y_hi = max(1, cy - r), min(n, cy + r)
        patch = terrain_cost[y_lo - 1:y_hi, x_lo - 1:x_hi]
        if patch.size > 0:
            mi = np.unravel_index(patch.argmin(), patch.shape)
            cluster_sol[2*k] = x_lo + mi[1]
            cluster_sol[2*k + 1] = y_lo + mi[0]
        else:
            cluster_sol[2*k], cluster_sol[2*k + 1] = cx, cy
    cluster_sol = deduplicate_bs(cluster_sol)

    # Population: 50% cluster-based, 25% mixed, 25% terrain-weighted
    pop = np.zeros((popSize, nVar), dtype=int)
    n_cluster = int(popSize * 0.5)
    n_mixed = int(popSize * 0.25)

    pop[0, :] = deduplicate_bs(cluster_sol.copy())
    for i in range(1, n_cluster):
        sol = cluster_sol.copy()
        for j in range(num_bs):
            sol[2*j] = int(np.clip(sol[2*j] + np.random.randint(-5, 6), 1, n))
            sol[2*j + 1] = int(np.clip(sol[2*j + 1] + np.random.randint(-5, 6), 1, n))
        pop[i, :] = deduplicate_bs(sol)

    for i in range(n_mixed):
        sol = np.zeros(nVar, dtype=int)
        half = num_bs // 2
        for j in range(half):
            sol[2*j] = int(np.clip(cluster_sol[2*j] + np.random.randint(-10, 11), 1, n))
            sol[2*j + 1] = int(np.clip(cluster_sol[2*j + 1] + np.random.randint(-10, 11), 1, n))
        indices = np.random.choice(n * n, size=num_bs - half, replace=False, p=weights)
        x_coords = (indices % n) + 1
        y_coords = (indices // n) + 1
        for j in range(half, num_bs):
            sol[2*j] = x_coords[j - half]
            sol[2*j + 1] = y_coords[j - half]
        pop[n_cluster + i, :] = deduplicate_bs(sol)

    for i in range(n_cluster + n_mixed, popSize):
        indices = np.random.choice(n * n, size=num_bs, replace=False, p=weights)
        x_coords = (indices % n) + 1
        y_coords = (indices // n) + 1
        sol = np.zeros(nVar, dtype=int)
        sol[0::2] = x_coords
        sol[1::2] = y_coords
        pop[i, :] = deduplicate_bs(sol)

    def evaluate(bs_x, bs_y):
        """Vectorized evaluation: coverage, avg SINR, and total cost."""
        bs_x = np.clip(bs_x, 1, n)
        bs_y = np.clip(bs_y, 1, n)
        key = tuple(np.r_[bs_x, bs_y])
        if key in eval_cache:
            return eval_cache[key]

        B = len(bs_x)
        U = num_users

        bx = bs_x - 1
        by = bs_y - 1
        bz = Z[by, bx]

        dux = ux[:, None] - bs_x[None, :]
        duy = uy[:, None] - bs_y[None, :]
        duz = uz[:, None] - bz[None, :]
        d = np.sqrt(dux**2 + duy**2 + duz**2)

        # LOS check
        n_los = 10
        xs = np.linspace(ux[:, None], bs_x[None, :], n_los, axis=-1).astype(int)
        ys = np.linspace(uy[:, None], bs_y[None, :], n_los, axis=-1).astype(int)
        xs = np.clip(xs, 1, n) - 1
        ys = np.clip(ys, 1, n) - 1
        terrain_line = Z[ys, xs]

        t = np.linspace(0, 1, n_los)
        los_h = uz[:, None, None] + t[None, None, :] * (bz[None, :, None] + 15 - uz[:, None, None])
        los = ~np.any(terrain_line > los_h, axis=-1)

        b_slope = slope[by, bx]
        u_slope = slope[uy0, ux0]
        theta = (u_slope[:, None] + b_slope[None, :]) / 2

        # Path loss: 2.2 for LOS, 3.8 for NLOS
        n_loss = np.where(los, 2.2, 3.8)
        PL = 32.4 + 10 * n_loss * np.log10(d + 1) + alpha * theta
        rx = Pt - PL

        rx_pow = 10 ** (rx / 10)
        best_pow = rx_pow.max(axis=1)
        sum_interf = rx_pow.sum(axis=1) - best_pow
        noise_pow = 10 ** (noise / 10)
        SINR_all = 10 * np.log10(best_pow / (sum_interf + noise_pow))

        covered = SINR_all > 5
        Coverage = covered.mean()
        AvgSINR = SINR_all.mean()
        Cost = terrain_cost[by, bx].sum()

        res = (Coverage, AvgSINR, Cost)
        eval_cache[key] = res
        return res

    marl_searcher = MARLSearcher(
        n_agents=num_bs, n=n, Z=Z, terrain_cost=terrain_cost, slope=slope,
        ux=ux, uy=uy, uz=uz, Cmin=Cmin,
        evaluate_fn=evaluate, deduplicate_fn=deduplicate_bs,
        max_action=25.0, gamma=0.95
    )
    MARL_INTERVAL = 10
    MARL_STEPS = 8
    MARL_TRAIN_PER_STEP = 2

    def fitness_with_adaptive_penalty(sol, t):
        """Adaptive penalty: high penalty early, gradually relaxed."""
        bs_x = sol[0::2]
        bs_y = sol[1::2]
        Coverage, AvgSINR, Cost = evaluate(bs_x, bs_y)

        P0 = 1e5
        P_min = 1e2
        t_clamped = min(t, MaxIt)
        P = P0 * (1 - t_clamped / MaxIt) + P_min

        epsilon = 0.02 * (1 - t_clamped / MaxIt)
        effective_min = Cmin - epsilon

        if Coverage < effective_min:
            deficit = effective_min - Coverage
            penalty = P * deficit * (1 + 10 * deficit)
        elif Coverage < Cmin:
            deficit = Cmin - Coverage
            penalty = P_min * deficit * deficit * 10
        else:
            penalty = 0
        return Cost + penalty, Coverage, AvgSINR, Cost

    def update_feasible_archive(archive, pop, cov_values, cost_values):
        for i in range(pop.shape[0]):
            if cov_values[i] >= Cmin:
                if not any(np.array_equal(item['sol'], pop[i]) for item in archive):
                    archive.append({'sol': pop[i].copy(), 'cov': cov_values[i], 'cost': cost_values[i]})
        archive.sort(key=lambda x: x['cost'])
        if len(archive) > 3 * popSize:
            del archive[3 * popSize:]
        return archive

    def coverage_margin_local_search(best_sol, K=40):
        """Local search on feasible solutions: trade coverage margin for cost reduction."""
        best_sol = deduplicate_bs(best_sol)
        bs_x = best_sol[0::2]
        bs_y = best_sol[1::2]
        best_cov, best_sinr, best_cost = evaluate(bs_x, bs_y)

        if best_cov < Cmin:
            return best_sol, best_cov, best_sinr, best_cost

        delta = best_cov - Cmin
        max_perturb = int(np.round(30 + 60 * delta))
        max_perturb = np.clip(max_perturb, 20, 80)

        for _ in range(K):
            new_sol = best_sol.copy()
            num_vars = np.random.randint(2, 6)
            idx = np.random.choice(nVar, num_vars, replace=False)
            for j in idx:
                new_sol[j] = np.clip(new_sol[j] + np.random.randint(-max_perturb, max_perturb+1), VarMin, VarMax)
            new_sol = deduplicate_bs(new_sol)
            nc, ns, nco = evaluate(new_sol[0::2], new_sol[1::2])
            if nc >= Cmin and nco < best_cost:
                best_sol, best_cov, best_sinr, best_cost = new_sol, nc, ns, nco
            elif nc >= Cmin and nco == best_cost and nc > best_cov:
                best_sol, best_cov, best_sinr, best_cost = new_sol, nc, ns, nco

        # Fine-tune each BS to local min cost
        for i in range(num_bs):
            ox, oy = best_sol[2*i], best_sol[2*i+1]
            best_tc = terrain_cost[oy-1, ox-1]
            bx, by = ox, oy
            for dx in range(-5,6):
                for dy in range(-5,6):
                    x2 = np.clip(ox+dx,1,n)
                    y2 = np.clip(oy+dy,1,n)
                    if terrain_cost[y2-1, x2-1] < best_tc:
                        best_tc = terrain_cost[y2-1, x2-1]
                        bx, by = x2, y2
            if bx != ox or by != oy:
                s = best_sol.copy()
                s[2*i], s[2*i+1] = bx, by
                s = deduplicate_bs(s)
                cc, _, cco = evaluate(s[0::2], s[1::2])
                if cc >= Cmin and cco < best_cost:
                    best_sol, best_cov, best_cost = s, cc, cco

        return best_sol, best_cov, best_sinr, best_cost

    def tournament_selection(pop, costs, k=5):
        cand = np.random.choice(len(pop), k, replace=False)
        return pop[cand[np.argmin(costs[cand])]].copy()

    def directed_migration(sol, archive):
        """Replace highest-cost BS with a BS from feasible archive."""
        if len(archive) < 3:
            return sol
        bs_x = sol[0::2]
        bs_y = sol[1::2]
        costs = terrain_cost[bs_y-1, bs_x-1]
        n_replace = np.random.randint(2, min(4, num_bs))
        worst = np.argsort(costs)[-n_replace:]
        g = archive[np.random.choice(len(archive))]['sol']
        gx, gy = g[0::2], g[1::2]
        out = sol.copy()
        for w in worst:
            j = np.random.randint(num_bs)
            out[2*w] = gx[j]
            out[2*w+1] = gy[j]
        return deduplicate_bs(out)

    def random_relocation(sol, prob=0.1):
        out = sol.copy()
        for i in range(num_bs):
            if np.random.rand() < prob:
                idx = np.random.choice(n*n, p=weights)
                out[2*i] = idx % n +1
                out[2*i+1] = idx //n +1
        return deduplicate_bs(out)

    def guided_crossover(pop, archive, costs, t):
        """Crossover guided by feasible archive solutions (probability increases over time)."""
        t_clamped = min(t, MaxIt)
        guide_prob = min(0.9, 0.3 + 0.6*t_clamped/MaxIt)
        if len(archive) >0 and np.random.rand() < guide_prob:
            ac = np.array([a['cost'] for a in archive])
            w = np.exp(-(ac-ac.min())/(ac.max()-ac.min()+1e-6))
            w /= w.sum()
            g = archive[np.random.choice(len(archive), p=w)]['sol']
            p1 = tournament_selection(pop, costs,5)
            p2 = tournament_selection(pop, costs,5)
            child = p1.copy()
            ng = int(np.ceil(nVar*(0.4+0.3*t_clamped/MaxIt)))
            idx = np.random.choice(nVar, ng, replace=False)
            child[idx] = g[idx]
            rest = [j for j in range(nVar) if j not in idx]
            child[rest] = np.where(np.random.rand(len(rest))<0.5, p1[rest], p2[rest])
        else:
            p1 = tournament_selection(pop,costs,5)
            p2 = tournament_selection(pop,costs,5)
            child = np.where(np.random.rand(nVar)<0.5, p1,p2)
        return deduplicate_bs(np.clip(child,1,n).astype(int))

    def terrain_guided_mutation(sol, mr):
        """Mutate toward lower-cost terrain regions."""
        if np.random.rand() > mr:
            return sol
        out = sol.copy()
        nv = np.random.randint(2,5)
        idx = np.random.choice(nVar, nv, replace=False)
        for j in idx:
            if np.random.rand() <0.5:
                c = out[j]
                r = int(np.clip(20+30*(1-mr),10,50))
                lo = max(1, c-r)
                hi = min(n, c+r)
                patch = terrain_cost[lo-1:hi-1, lo-1:hi-1]
                if patch.size>0:
                    lw = 1/(patch.ravel()+1e-6)
                    lw /= lw.sum()
                    li = np.random.choice(len(lw), p=lw)
                    sz = hi-lo
                    if j%2==0:
                        out[j] = lo + li%sz
                    else:
                        out[j] = lo + li//sz
                else:
                    out[j] = np.random.randint(1,n+1)
            else:
                out[j] = np.random.randint(1,n+1)
        return deduplicate_bs(out)

    def feasibility_repair(sol):
        """Try to repair near-feasible solutions by moving toward better coverage."""
        sol = deduplicate_bs(sol)
        cov,_,_ = evaluate(sol[0::2], sol[1::2])
        if cov >= Cmin or cov < Cmin-0.05:
            return sol
        best = sol.copy()
        bc = cov
        for i in range(num_bs):
            ox, oy = best[2*i], best[2*i+1]
            for dx in (-3,-2,-1,1,2,3):
                for dy in (-3,-2,-1,1,2,3):
                    x2 = np.clip(ox+dx,1,n)
                    y2 = np.clip(oy+dy,1,n)
                    s = best.copy()
                    s[2*i], s[2*i+1] = x2,y2
                    s = deduplicate_bs(s)
                    c,_,_ = evaluate(s[0::2], s[1::2])
                    if c>bc:
                        best, bc = s, c
                        if bc>=Cmin:
                            return best
        return best

    # Main loop
    start_time = time.time()

    bestCost = np.inf
    bestSol = None
    bestFeasibleSol = None
    bestFeasibleCost = np.inf

    curve = np.zeros(MAX_TIME)
    coverage_history = np.zeros(MAX_TIME)
    sinr_history = np.zeros(MAX_TIME)
    cost_history = np.zeros(MAX_TIME)

    feasible_archive = []
    stagnation_counter = 0
    last_recorded_second = -1
    it = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= MAX_TIME:
            break

        costs = np.zeros(popSize)
        cov_v = np.zeros(popSize)
        sinr_v = np.zeros(popSize)
        cost_v = np.zeros(popSize)

        for i in range(popSize):
            costs[i], cov_v[i], sinr_v[i], cost_v[i] = fitness_with_adaptive_penalty(pop[i], it)

        idx = np.argsort(costs)
        pop, costs, cov_v, sinr_v, cost_v = pop[idx], costs[idx], cov_v[idx], sinr_v[idx], cost_v[idx]

        fm = cov_v >= Cmin
        if fm.any():
            fi = np.where(fm)[0][0]
            if cost_v[fi] < bestFeasibleCost:
                bestFeasibleCost = cost_v[fi]
                bestFeasibleSol = pop[fi].copy()

        if costs[0] < bestCost:
            bestCost = costs[0]
            bestSol = pop[0].copy()
            stagnation_counter = 0
        else:
            stagnation_counter +=1

        if bestFeasibleSol is not None:
            cov, sinr, cst = evaluate(bestFeasibleSol[0::2], bestFeasibleSol[1::2])
        else:
            cov, sinr, cst = evaluate(bestSol[0::2], bestSol[1::2])

        # Record per second
        current_second = min(int(elapsed), MAX_TIME - 1)
        while last_recorded_second < current_second:
            last_recorded_second += 1
            sec = last_recorded_second
            curve[sec] = bestFeasibleCost if bestFeasibleSol is not None else bestCost
            coverage_history[sec] = cov
            sinr_history[sec] = sinr
            cost_history[sec] = cst
            print(f'Run {run_id} | Time {sec + 1}/{MAX_TIME} | Cost = {curve[sec]:.2f} | Cov = {cov*100:.1f}%')

        feasible_archive = update_feasible_archive(feasible_archive, pop, cov_v, cost_v)

        if bestFeasibleSol is not None and it %5 ==0:
            K = int(20+30*it/MaxIt)
            bestFeasibleSol, cov, sinr, cst = coverage_margin_local_search(bestFeasibleSol, K)
            if cst < bestFeasibleCost:
                bestFeasibleCost = cst

        # MARL-enhanced local search every MARL_INTERVAL generations
        if bestFeasibleSol is not None and (it+1) % MARL_INTERVAL == 0:
            marl_best = bestFeasibleSol.copy()
            cur_cov, _, cur_cost = evaluate(marl_best[0::2], marl_best[1::2])
            for s in range(MARL_STEPS):
                new_sol, new_cov, new_cost, _ = marl_searcher.step(
                    marl_best, it, MaxIt, cur_cov, cur_cost
                )
                if new_cov >= Cmin and new_cost < cur_cost:
                    marl_best = new_sol
                    cur_cov, cur_cost = new_cov, new_cost
                for _ in range(MARL_TRAIN_PER_STEP):
                    marl_searcher.train()
            _, _, marl_cost = evaluate(marl_best[0::2], marl_best[1::2])
            if marl_cost < bestFeasibleCost:
                bestFeasibleSol = marl_best
                bestFeasibleCost = marl_cost
                cov, sinr, cst = evaluate(bestFeasibleSol[0::2], bestFeasibleSol[1::2])

        # Restart on stagnation
        if stagnation_counter >=8:
            ni = int(popSize*0.5)
            imm = np.random.randint(1,n+1,(ni,nVar))
            for i in range(ni):
                imm[i] = deduplicate_bs(imm[i])
            if bestFeasibleSol is not None:
                for i in range(min(5,ni)):
                    imm[i] = bestFeasibleSol.copy()
                    vs = np.random.choice(nVar, np.random.randint(3,6), replace=False)
                    for j in vs:
                        v = bestFeasibleSol[j]
                        imm[i,j] = np.random.randint(max(1,v-50), min(n, v+50)+1)
                    imm[i] = deduplicate_bs(imm[i])
            pop[-ni:] = imm
            stagnation_counter =0

        # Generate new population
        newPop = np.zeros_like(pop)
        elite = min(3, fm.sum())
        if elite>0:
            newPop[:elite] = pop[np.where(fm)[0][:elite]]
        else:
            newPop[0] = pop[0]
        if bestFeasibleSol is not None:
            newPop[0] = bestFeasibleSol

        for i in range(elite, popSize):
            child = pop[i].copy()
            if len(feasible_archive)>0 and np.random.rand() < 0.3+0.4*it/MaxIt:
                child = directed_migration(child, feasible_archive)
            if np.random.rand()<0.7:
                child = guided_crossover(pop, feasible_archive, costs, it)
            mr = 0.4 if stagnation_counter>=5 else 0.2
            child = terrain_guided_mutation(child, mr)
            if np.random.rand()<0.05:
                child = random_relocation(child, 0.15)
            child = feasibility_repair(child)
            newPop[i] = child
        pop = newPop

        it += 1

    # Fill remaining seconds
    while last_recorded_second < MAX_TIME - 1:
        last_recorded_second += 1
        sec = last_recorded_second
        curve[sec] = bestFeasibleCost if bestFeasibleSol is not None else bestCost
        coverage_history[sec] = cov
        sinr_history[sec] = sinr
        cost_history[sec] = cst
        print(f'Run {run_id} | Time {sec + 1}/{MAX_TIME} | Cost = {curve[sec]:.2f} | Cov = {cov*100:.1f}%')

    # Post-processing
    if bestFeasibleSol is not None:
        print(f"Run {run_id} final local search...")
        for _ in range(30):
            bestFeasibleSol, _, _, _ = coverage_margin_local_search(bestFeasibleSol, K=50)

        cov_check, _, _ = evaluate(bestFeasibleSol[0::2], bestFeasibleSol[1::2])
        if cov_check < Cmin:
            print(f"Warning: final best solution has coverage {cov_check*100:.1f}%, below {Cmin*100}%. Using best from archive.")
            if len(feasible_archive) > 0:
                bestFeasibleSol = feasible_archive[0]['sol'].copy()
            else:
                print("Archive is empty, keeping current solution (may be infeasible).")

    if bestFeasibleSol is not None:
        bs_x, bs_y = bestFeasibleSol[0::2], bestFeasibleSol[1::2]
    else:
        bs_x, bs_y = bestSol[0::2], bestSol[1::2]

    Coverage, AvgSINR, Cost = evaluate(bs_x, bs_y)
    elapsed = time.time() - start_time

    print('\n'+'='*50)
    print(f'RUN {run_id} FINAL RESULT')
    print('='*50)
    print(f'Coverage = {Coverage*100:.2f}%')
    print(f'AvgSINR  = {AvgSINR:.2f} dB')
    print(f'Cost     = {Cost:.2f}')
    print(f'Time     = {elapsed:.2f} s')
    print('='*50)

    np.save(f'best_solution_run{run_id}.npy', bestFeasibleSol if bestFeasibleSol is not None else bestSol)

    # Save CSV
    output_data = []
    for sec in range(MAX_TIME):
        output_data.append({
            'Second': sec + 1,
            'Coverage': coverage_history[sec],
            'AvgSINR': sinr_history[sec],
            'Cost': cost_history[sec],
            'BestCost': curve[sec]
        })
    csv_filename = f'ga_results_run{run_id}.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Second', 'Coverage', 'AvgSINR', 'Cost', 'BestCost'])
        writer.writeheader()
        writer.writerows(output_data)
    print(f'CSV data saved to {csv_filename}')

    # Convergence curve
    plt.figure(figsize=(10,6))
    plt.plot(curve, lw=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Best Cost')
    plt.title(f'GA Convergence (Run {run_id})')
    plt.grid(alpha=0.3)
    final_best = bestFeasibleCost if bestFeasibleSol is not None else bestCost
    plt.text(0.7, 0.2, f'Best Cost = {final_best:.2f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'ga_convergence_run{run_id}.png', dpi=300)
    plt.close()

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='turbo', edgecolor='none', alpha=0.8)
    bs_z = Z[bs_y-1, bs_x-1] + 15
    ax.scatter(bs_x, bs_y, bs_z, color='red', marker='^', s=100, label='Base Stations')
    ax.set_title(f'BS Deployment 3D (Run {run_id})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text2D(0.05, 0.95, f'Cost = {Cost:.2f}\nCoverage = {Coverage*100:.1f}%',
              transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(f'bs_deployment_run{run_id}.png', dpi=300)
    plt.close()

    # Top-down view
    plt.figure(figsize=(10, 8))
    contourf = plt.contourf(X, Y, Z, levels=20, cmap='terrain')
    plt.colorbar(contourf, label='Elevation')
    plt.contour(X, Y, Z, levels=15, colors='dimgray', linewidths=2, alpha=0.6)
    plt.scatter(bs_x, bs_y, color='red', marker='^', s=100, label='Base Stations')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('BS Deployment Top View')
    plt.legend()
    plt.text(0.05, 0.95, f'Cost = {Cost:.2f}\nCoverage = {Coverage*100:.1f}%',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'bs_xy_run{run_id}.png', dpi=300)
    plt.close()

    iter_records = []
    for sec in range(MAX_TIME):
        iter_records.append({
            'Run': run_id,
            'Second': sec + 1,
            'Coverage': coverage_history[sec],
            'AvgSINR': sinr_history[sec],
            'Cost': cost_history[sec],
            'BestCost': curve[sec]
        })
    bs_coords = [(bs_x[i], bs_y[i]) for i in range(num_bs)]
    summary = {
        'Run': run_id,
        'Coverage': Coverage,
        'AvgSINR': AvgSINR,
        'Cost': Cost,
        'BestCost': final_best
    }
    return iter_records, bs_coords, summary


if __name__ == '__main__':
    num_runs = 20
    MAX_TIME = 200

    all_iter_data = []
    all_bs_coords = []
    all_summary = []

    all_data = {
        'Coverage': np.zeros((MAX_TIME, num_runs)),
        'AvgSINR': np.zeros((MAX_TIME, num_runs)),
        'Cost': np.zeros((MAX_TIME, num_runs)),
        'BestCost': np.zeros((MAX_TIME, num_runs))
    }

    for run_idx in range(num_runs):
        run_id = run_idx + 1
        iter_data, bs_coords, summary = run_simulation(run_id)
        all_iter_data.extend(iter_data)
        all_bs_coords.append((run_id, bs_coords))
        all_summary.append(summary)
        for item in iter_data:
            sec_idx = item['Second'] - 1
            all_data['Coverage'][sec_idx, run_idx] = item['Coverage']
            all_data['AvgSINR'][sec_idx, run_idx] = item['AvgSINR']
            all_data['Cost'][sec_idx, run_idx] = item['Cost']
            all_data['BestCost'][sec_idx, run_idx] = item['BestCost']

    # Aggregate Excel
    wb = Workbook()

    for param_name, data in all_data.items():
        ws = wb.create_sheet(param_name)
        header = ['Second']
        for r in range(num_runs):
            header.append(f'Run{r+1}')
        ws.append(header)
        for sec in range(MAX_TIME):
            row = [sec + 1]
            for r in range(num_runs):
                row.append(data[sec, r])
            ws.append(row)
        ws.append([])
        ws.append(['Statistics'])
        mean_row = ['Mean']
        std_row = ['Std']
        min_row = ['Min']
        max_row = ['Max']
        final_row = ['Final']
        for r in range(num_runs):
            mean_row.append(np.mean(data[:, r]))
            std_row.append(np.std(data[:, r]))
            min_row.append(np.min(data[:, r]))
            max_row.append(np.max(data[:, r]))
            final_row.append(data[-1, r])
        ws.append(mean_row)
        ws.append(std_row)
        ws.append(min_row)
        ws.append(max_row)
        ws.append(final_row)
        ws.column_dimensions['A'].width = 12
        for col_idx in range(1, num_runs + 2):
            if col_idx < 26:
                col_letter = chr(65 + col_idx)
            else:
                col_letter = chr(64 + col_idx // 26) + chr(65 + col_idx % 26)
            ws.column_dimensions[col_letter].width = 16

    if "Sheet" in wb.sheetnames:
        wb.remove(wb['Sheet'])

    # Summary sheet
    ws_summary = wb.create_sheet('Summary')
    ws_summary.append(['Run', 'Final Coverage', 'Final AvgSINR', 'Final Cost', 'Final BestCost'])
    for s in all_summary:
        ws_summary.append([
            s['Run'],
            s['Coverage'],
            s['AvgSINR'],
            s['Cost'],
            s['BestCost']
        ])
    ws_summary.append([])
    ws_summary.append(['Statistics', 'Coverage', 'AvgSINR', 'Cost', 'BestCost'])
    cov_list = [s['Coverage'] for s in all_summary]
    sinr_list = [s['AvgSINR'] for s in all_summary]
    cost_list = [s['Cost'] for s in all_summary]
    bestcost_list = [s['BestCost'] for s in all_summary]
    ws_summary.append([
        'Mean',
        np.mean(cov_list),
        np.mean(sinr_list),
        np.mean(cost_list),
        np.mean(bestcost_list)
    ])
    ws_summary.append([
        'Std',
        np.std(cov_list),
        np.std(sinr_list),
        np.std(cost_list),
        np.std(bestcost_list)
    ])
    ws_summary.append([
        'Min',
        np.min(cov_list),
        np.min(sinr_list),
        np.min(cost_list),
        np.min(bestcost_list)
    ])
    ws_summary.append([
        'Max',
        np.max(cov_list),
        np.max(sinr_list),
        np.max(cost_list),
        np.max(bestcost_list)
    ])

    # Final BS sheet
    ws_bs = wb.create_sheet('Final_BS')
    ws_bs.append(['Run', 'BS_ID', 'X', 'Y'])
    for run_id, coords in all_bs_coords:
        for i, (x, y) in enumerate(coords, start=1):
            ws_bs.append([run_id, i, x, y])

    save_name = 'all_ga_results.xlsx'
    wb.save(save_name)
    print(f'\nMulti-run simulation results saved to {save_name}')
    print('Sheets: Coverage, AvgSINR, Cost, BestCost, Summary, Final_BS')

    # Save all BS coordinates
    with open('final_bs_coordinates.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'BS_ID', 'X', 'Y', 'Coverage', 'Cost'])
        for (run_id, coords), s in zip(all_bs_coords, all_summary):
            for i, (x, y) in enumerate(coords, start=1):
                writer.writerow([run_id, i, x, y, s['Coverage'], s['Cost']])
    print('\nBS coordinates of all runs saved to final_bs_coordinates.csv')

    # Find and save overall best feasible solution
    feasible = [s for s in all_summary if s['Coverage'] >= 0.91]
    if feasible:
        best_overall = min(feasible, key=lambda s: s['Cost'])
        best_run_id = best_overall['Run']
        best_coords = all_bs_coords[best_run_id - 1][1]
        print(f'\nOverall best solution: Run {best_run_id}, Coverage={best_overall["Coverage"]*100:.2f}%, Cost={best_overall["Cost"]:.2f}')

        with open('best_overall_bs_coordinates.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['BS_ID', 'X', 'Y'])
            for i, (x, y) in enumerate(best_coords, start=1):
                writer.writerow([i, x, y])
            writer.writerow([])
            writer.writerow(['Coverage', best_overall['Coverage']])
            writer.writerow(['AvgSINR', best_overall['AvgSINR']])
            writer.writerow(['Cost', best_overall['Cost']])
            writer.writerow(['BestCost', best_overall['BestCost']])
            writer.writerow(['From Run', best_run_id])

        best_sol_path = f'best_solution_run{best_run_id}.npy'
        best_overall_sol = np.load(best_sol_path)
        np.save('best_overall_solution.npy', best_overall_sol)
        print('Overall best BS coordinates saved to best_overall_bs_coordinates.csv')
        print('Overall best solution vector saved to best_overall_solution.npy')
    else:
        print('\nWarning: no feasible solution with Coverage >= 91% found in 20 runs!')
