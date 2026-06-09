import pulp


def solve_linearized_problem(M, N, T, V_m, P_m, S, K_n, T_pred, tau_f, tau_transmission, 
                             tau_waiting, tau_computing, l_z, r_n, z_mi):

    prob = pulp.LpProblem("Linearized_Task_Offloading", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", [(t,m,i,n) for t in T for m in M for i in V_m[m] for n in N], cat='Binary')
    c = pulp.LpVariable.dicts("c", [(t,s,n) for t in T for s in S for n in N], cat='Binary')
    y = pulp.LpVariable.dicts("y", [(t,m,i,j,n,n_prime) for t in T for m in M for i in V_m[m] for j in P_m.get((m,i),[]) for n in N for n_prime in N], cat='Binary')

    T_mi_n = pulp.LpVariable.dicts("T", [(t,m,i,n) for t in T for m in M for i in V_m[m] for n in N], lowBound=0)

    # Objective
    prob += (1/len(M))*pulp.lpSum(x[t,m,V_m[m][-1],n]*T_mi_n[t,m,V_m[m][-1],n] for t in T for m in M for n in N)

    # Constraints
    for t in T:
        for m in M:
            for i in V_m[m]:
                for j in P_m.get((m,i),[]):
                    prob += (pulp.lpSum(x[t,m,i,n]*T_pred[t,m,i,n] for n in N) >=
                             pulp.lpSum(y[t,m,i,j,n,n_prime]*(T_mi_n[t,m,j,n_prime]+tau_f[n_prime,n,m,j]) for n in N for n_prime in N))

                    for n in N:
                        for n_prime in N:
                            prob += y[t,m,i,j,n,n_prime] <= x[t,m,i,n]
                            prob += y[t,m,i,j,n,n_prime] <= x[t,m,j,n_prime]
                            prob += y[t,m,i,j,n,n_prime] >= x[t,m,i,n] + x[t,m,j,n_prime] - 1

    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    prob += T_mi_n[t,m,i,n] == T_pred[t,m,i,n]+tau_transmission[t,m,i,n]+tau_waiting[t,m,i,n]+tau_computing[t,m,i,n]+(1-c[t,z_mi[m,i],n])*(l_z[z_mi[m,i]]/r_n[n])

    for t in T:
        for n in N:
            prob += pulp.lpSum(c[t,s,n] for s in S) <= K_n[n]

    for t in T:
        for m in M:
            for i in V_m[m]:
                prob += pulp.lpSum(x[t,m,i,n] for n in N) == 1

    prob.solve()

    return prob.status, pulp.value(prob.objective)

# Pass your data and solve
