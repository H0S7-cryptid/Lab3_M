import random
import numpy as np


class ProductionSimulator:
    def __init__(
        self,
        y_A0=120,
        y_B0=150,
        y_CA0=20,
        y_CB0=30,
        y_11_0=40,
        P_A=40,
        P_B=60,
        P_C=None,
        D11=10,
        D21=2,
        D22=6,
        D23=10,
        X1=15,
        threshold_critical=0.2,
        threshold_stop=0.05,
        replenishment_levels=None,
        replenishment_probs=None,
    ):
        self.y_A0 = float(y_A0)
        self.y_B0 = float(y_B0)
        self.y_CA0 = float(y_CA0)
        self.y_CB0 = float(y_CB0)
        self.y_11_0 = float(y_11_0)
        self.P_A = float(P_A)
        self.P_B = float(P_B)
        self.P_C = float(P_C) if P_C is not None else min(self.P_A, self.P_B)
        self.D11 = float(D11)
        self.D21 = float(D21)
        self.D22 = float(D22)
        self.D23 = float(D23)
        self.X1 = float(X1)
        self.threshold_critical = float(threshold_critical)
        self.threshold_stop = float(threshold_stop)
        self.replenishment_levels = (
            [0.0, 0.5, 0.7, 0.4]
            if replenishment_levels is None
            else list(replenishment_levels)
        )
        self.replenishment_probs = (
            [0.4, 0.2, 0.1, 0.3]
            if replenishment_probs is None
            else list(replenishment_probs)
        )

    def sample_replenishment(self, rng):
        return rng.choices(self.replenishment_levels, weights=self.replenishment_probs, k=1)[0]

    def simulate(self, T=100, dt=1, seed=None):
        steps = int(round(T / dt)) + 1
        times = np.linspace(0, T, steps)

        y_A = np.zeros(steps)
        y_B = np.zeros(steps)
        pA = np.zeros(steps)
        pB1 = np.zeros(steps)
        pB2 = np.zeros(steps)
        pB3 = np.zeros(steps)
        y_CA = np.zeros(steps)
        y_CB = np.zeros(steps)
        G = np.zeros(steps)
        fA = np.zeros(steps)
        fB = np.zeros(steps)
        g = np.zeros(steps)

        y_A[0] = self.y_A0
        y_B[0] = self.y_B0
        pA[0] = self.y_11_0
        pB1[0] = 40.0
        pB2[0] = 40.0
        pB3[0] = 40.0
        y_CA[0] = self.y_CA0
        y_CB[0] = self.y_CB0
        G[0] = 0.0

        rng = random.Random(seed)

        for k in range(steps - 1):
            current_y_A = y_A[k]
            current_y_B = y_B[k]
            current_pA = pA[k]
            current_pB1 = pB1[k]
            current_pB2 = pB2[k]
            current_pB3 = pB3[k]
            current_y_CA = y_CA[k]
            current_y_CB = y_CB[k]
            current_G = G[k]

            addition_A = self.sample_replenishment(rng) * self.y_A0
            addition_B = self.sample_replenishment(rng) * self.y_B0
            current_y_A += addition_A + self.X1
            current_y_B += addition_B

            if current_y_A <= self.threshold_stop * self.y_A0:
                input_A = 0.0
            else:
                input_A = min(self.P_A, current_y_A)
            if current_y_B <= self.threshold_stop * self.y_B0:
                input_B = 0.0
            else:
                input_B = min(self.P_B, current_y_B)

            output_A = current_pA / self.D11
            output_B1 = current_pB1 / self.D21
            output_B2 = current_pB2 / self.D22
            output_B3 = current_pB3 / self.D23

            possible_assembly = min(current_y_CA, current_y_CB, self.P_C)
            assembly_rate = possible_assembly

            next_y_A = max(current_y_A - input_A * dt, 0.0)
            next_y_B = max(current_y_B - input_B * dt, 0.0)
            next_pA = max(current_pA + (input_A - output_A) * dt, 0.0)
            next_pB1 = max(current_pB1 + (input_B - output_B1) * dt, 0.0)
            next_pB2 = max(current_pB2 + (output_B1 - output_B2) * dt, 0.0)
            next_pB3 = max(current_pB3 + (output_B2 - output_B3) * dt, 0.0)
            next_y_CA = max(current_y_CA + (output_A - assembly_rate) * dt, 0.0)
            next_y_CB = max(current_y_CB + (output_B3 - assembly_rate) * dt, 0.0)
            next_G = current_G + assembly_rate * dt

            y_A[k + 1] = next_y_A
            y_B[k + 1] = next_y_B
            pA[k + 1] = next_pA
            pB1[k + 1] = next_pB1
            pB2[k + 1] = next_pB2
            pB3[k + 1] = next_pB3
            y_CA[k + 1] = next_y_CA
            y_CB[k + 1] = next_y_CB
            G[k + 1] = next_G
            fA[k] = input_A
            fB[k] = input_B
            g[k] = assembly_rate

        fA[-1] = fA[-2] if steps > 1 else 0.0
        fB[-1] = fB[-2] if steps > 1 else 0.0
        g[-1] = g[-2] if steps > 1 else 0.0

        return {
            "time": times,
            "y_A": y_A,
            "y_B": y_B,
            "pA": pA,
            "pB1": pB1,
            "pB2": pB2,
            "pB3": pB3,
            "y_CA": y_CA,
            "y_CB": y_CB,
            "G": G,
            "fA": fA,
            "fB": fB,
            "g": g,
        }

    def calculate_metrics(self, result):
        T = result["time"][-1] if len(result["time"]) > 1 else 1.0
        threshold_A = self.threshold_stop * self.y_A0
        threshold_B = self.threshold_stop * self.y_B0
        stopA = result["y_A"] <= threshold_A
        stopB = result["y_B"] <= threshold_B
        downtime_frac_A = float(np.mean(stopA))
        downtime_frac_B = float(np.mean(stopB))
        stop_prob_A = float(np.any(stopA))
        stop_prob_B = float(np.any(stopB))

        G_total = float(result["G"][-1])
        q_avg = float(G_total / T)
        I_avg_A = float(np.mean(result["y_A"]))
        I_avg_B = float(np.mean(result["y_B"]))

        return {
            "G_total": G_total,
            "q_avg": q_avg,
            "I_avg_A": I_avg_A,
            "I_avg_B": I_avg_B,
            "downtime_frac_A": downtime_frac_A,
            "downtime_frac_B": downtime_frac_B,
            "stop_prob_A": stop_prob_A,
            "stop_prob_B": stop_prob_B,
        }
