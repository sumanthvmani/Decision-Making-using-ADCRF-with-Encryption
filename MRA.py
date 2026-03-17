import numpy as np
import time


def bounds(s, lb, ub):
    return np.clip(s, lb, ub)


def mu_inv(y, mu):
    return (((1 + mu) ** np.abs(y) - 1) / mu) * np.sign(y)


def MRA(v, fobj, lb, ub, T_max):
    # Mud Ring Algorithm (MRA)
    SearchAgents_no, dim = v.shape
    vLb = 0.6 * lb
    vUb = 0.6 * ub
    MRLeader_pos = np.zeros(dim)
    MRLeader_score = np.inf
    Convergence_curve = np.zeros(T_max)
    start_time = time.time()
    t = 0
    while t < T_max:
        for i in range(SearchAgents_no):
            v[i, :] = bounds(v[i, :], lb[i, :], ub[i, :])
            v[i, :] = np.random.rand(dim)

            fitness = fobj(v[i, :])
            if fitness < MRLeader_score:
                MRLeader_score = fitness
                MRLeader_pos = v[i, :].copy()
        a = 2 * (1 - t / T_max)
        for i in range(SearchAgents_no):
            r = np.random.rand()
            K = 2 * a * r - a
            C = 2 * r
            l = np.random.rand()
            for j in range(dim):
                if abs(K) >= 1:
                    v[i, :] = bounds(v[i, :], vLb[i, :], vUb[i, :])
                    v[i, j] += v[i, j]
                else:
                    A = abs(C * MRLeader_pos[j] - v[i, j])
                    pos = MRLeader_pos[j] * np.sin(l * 2 * np.pi) - K * A
                    v[i, j] = mu_inv(bounds(pos, lb[i, j], ub[i, j]), np.random.rand())
        Convergence_curve[t] = MRLeader_score
        t += 1
    elapsed_time = time.time() - start_time
    return MRLeader_score, Convergence_curve, MRLeader_pos, elapsed_time
