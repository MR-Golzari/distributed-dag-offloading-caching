import random
# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from input import INPUT_DICT
from input import learning_arg as learning_arguments
from input import GCN_paramaters
from simulator import MEC_Simulator
from datetime import datetime
import sys
from tqdm import tqdm
import os
from pathlib import Path

from telegram import Telegram
import json
import pulp
from utils import update_single_user_position

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random

import pulp
import csv

def solve_linearized_problem(M, N, T, V_m, P_m, S, K_n,i,o,z_mi,between_server_costs, tau_transmission, 
                             tau_waiting, tau_computing, l_z, r_cloud, U, lasttasks):

    # Define the MILP problem
    prob = pulp.LpProblem("Linearized_Task_Offloading", pulp.LpMinimize)

    # Decision Variables
    x = pulp.LpVariable.dicts("x", [(t, m, i, n) for t in T for m in M for i in V_m[m] for n in N], cat='Binary')
    c = pulp.LpVariable.dicts("c", [(t, s, n) for t in T for s in S for n in N], cat='Binary')
    y_cloud = pulp.LpVariable.dicts("y_cloud", [(t, m, i, n) for t in T for m in M for i in V_m[m] for n in N], cat='Binary')
    y = pulp.LpVariable.dicts("y", [(t, m, i, n, n_prime) for t in T for m in M for i in V_m[m] for n in N for n_prime in N], cat='Binary')
    w = pulp.LpVariable.dicts("w", [(t, m, i, j, n, n_prime) for t in T for m in M for i in V_m[m] for j in P_m.get((m, i), []) for n in N for n_prime in N], cat='Binary')

    T_pred = pulp.LpVariable.dicts("T_pred", [(t, m, i, n) for t in T for m in M for i in V_m[m] for n in N], lowBound=0)
    T_assign = pulp.LpVariable.dicts("T_assign", [(t, m, i, n) for t in T for m in M for i in V_m[m] for n in N], lowBound=0)
    tau_service = pulp.LpVariable.dicts("tau_service", [(t, m, i, n) for t in T for m in M for i in V_m[m] for n in N], lowBound=0)

    # Objective Function: Minimize Task Completion Time
    prob += (1.0 / len(M)/len(T)) * pulp.lpSum(T_assign[t, m, lasttasks[m], n] for t in T for m in M for n in N)

    # Task Precedence Constraint (Linearized)
    for t in T:
        for m in M:
            for i in V_m[m]:
                for j in P_m.get((m, i), []):
                    prob += (pulp.lpSum(T_pred[t, m, i, n] for n in N) >=
                             pulp.lpSum(T_assign[t, m, j, n_prime] for n_prime in N) + pulp.lpSum(between_server_costs[n_prime, n]*o[m,j] * w[t, m, i, j, n, n_prime]
                                        for n in N for n_prime in N))

    # Linearization of T_pred
    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    prob += T_pred[t, m, i, n] <= U * x[t, m, i, n]

    # Linearization of T_assign
    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    prob += T_assign[t, m, i, n] >= T_pred[t, m, i, n] + tau_transmission[t,m,i,n] + tau_waiting[n] + tau_computing[m,i,n] + tau_service[t, m, i, n] - U * (1 - x[t, m, i, n])
                    prob += T_assign[t, m, i, n] <= U * x[t, m, i, n]

    # Linearization Constraints for w
    for t in T:
        for m in M:
            for i in V_m[m]:
                for j in P_m.get((m, i), []):
                    for n in N:
                        for n_prime in N:
                            prob += w[t, m, i, j, n, n_prime] <= x[t, m, i, n]
                            prob += w[t, m, i, j, n, n_prime] <= x[t, m, j, n_prime]
                            prob += w[t, m, i, j, n, n_prime] >= x[t, m, i, n] + x[t, m, j, n_prime] - 1

    # Service Latency
    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    prob += tau_service[t, m, i, n] == y_cloud[t, m, i, n] * (l_z[z_mi[m, i]] / r_cloud[n]) + \
                            pulp.lpSum(y[t, m, i, n, n_prime] * between_server_costs[n_prime,n]*l_z[z_mi[m, i]] for n_prime in N)

    # Single Service Source Constraint
    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    prob += y_cloud[t, m, i, n] + pulp.lpSum(y[t, m, i, n, n_prime] for n_prime in N) == 1

    # Cache Availability Constraint
    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    for n_prime in N:
                        prob += y[t, m, i, n, n_prime] <= c[t, z_mi[m, i], n_prime]

    # Cloudlet Selection Constraint
    for t in T:
        for m in M:
            for i in V_m[m]:
                for n in N:
                    prob += y_cloud[t, m, i, n] >= 1 - pulp.lpSum(c[t, z_mi[m, i], n_prime] for n_prime in N)

    # Cache Capacity Constraint
    for t in T:
        for n in N:
            prob += pulp.lpSum(c[t, s, n] for s in S) <= K_n[n]

    # Server Assignment Constraint
    for t in T:
        for m in M:
            for i in V_m[m]:
                prob += pulp.lpSum(x[t, m, i, n] for n in N) == 1

    # Solve the MILP problem
    print("Solving the MILP problem...")
    prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=60))

    return prob,prob.status, pulp.value(prob.objective)


def compute_tau_values(T,tasks, simulator):
    tau_transmission = {}
    tau_waiting = {}
    tau_computing = {}
    distances={}
# Define random latencies
    for t in T:
        for m, task_list in tasks.items():
            simulator.users[m].pos = update_single_user_position(simulator.xlim,simulator.ylim,simulator.users[m].pos, simulator.users[m].simulator.user_directions[simulator.users[m].id], simulator.street_positions_x, simulator.street_positions_y, simulator.vmax, simulator.vmin, simulator.street_width, t)
            gateway = simulator.users[m].find_nearest_server(simulator.servers)

            for i, task in task_list.items():
                for n, server in simulator.servers.items():
                    # Transmission latency (assumed pre-calculated)
                    tau_transmission[(t,m, i, n)] = task.input_data_length* ((1.0/ simulator.users[m].rate_to_gateway)+simulator.between_server_costs[gateway,n])
    for m, task_list in tasks.items():
        for i, task in task_list.items():
            for n, server in simulator.servers.items():
                # Computing latency (CPU cycles required / server frequency)
                computing_latency = task.cpu_cycle / server.frequency
                tau_computing[(m, i, n)] = computing_latency

    for n, server in simulator.servers.items():
                # Waiting latency (based on server load)
                waiting_latency = server.load * (1e6) / server.frequency
                tau_waiting[(n)] = waiting_latency

    return tau_transmission, tau_waiting, tau_computing

if __name__ == '__main__':
    outputpath = 'results'
    telegram = Telegram()
    arg_str=""
    args = sys.argv[1:]
    file_id=""
    argstr2 = "nuser"
    while(len(args)>=2):
        arg1=args.pop(0)
        arg2=args.pop(0)
        if arg1 == '-comment':
            INPUT_DICT['comment'] = arg2
            print('comment=',INPUT_DICT['comment'])
        if arg1 == '-folder':
            INPUT_DICT['Folder'] = arg2
            print('folder=',INPUT_DICT['Folder'])
        if arg1 == '-file_id':
            file_id = arg2
            print('file_id=',file_id)
            arg_str = arg_str +arg2+'_'

        if arg1 == '-nserver':
            INPUT_DICT['Number of servers'] = int(arg2)
            print('Number of servers=',INPUT_DICT['Number of servers'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-nservice':
            INPUT_DICT['Number of services'] = int(arg2)
            print('Number of services=',INPUT_DICT['Number of services'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-nuser':
            INPUT_DICT['Number of users'] = int(arg2)
            print('Number of users=',INPUT_DICT['Number of users'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-nepisode':
            INPUT_DICT['Number of episodes'] = int(arg2)
            print('Number of episodes=',INPUT_DICT['Number of episodes'])
            #arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-ntasks':
            INPUT_DICT['Number of tasks for each user'] = int(arg2)
            print('Number of episodes=',INPUT_DICT['Number of tasks for each user'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-max_explore':
            learning_arguments['maximum_exploration'] = int(arg2)
            print('maximum_exploration=',learning_arguments['maximum_exploration'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-min_service_data_length':
            INPUT_DICT['min_service_data_length'] = float(arg2)
            print('min_service_data_length=',INPUT_DICT['min_service_data_length'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-max_service_data_length':
            INPUT_DICT['max_service_data_length'] = float(arg2)
            print('max_service_data_length=',INPUT_DICT['max_service_data_length'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-max_data_length':
            INPUT_DICT['max_data_length'] = float(arg2)
            print('max_data_length=',INPUT_DICT['max_data_length'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-learning_rate':
            learning_arguments['learning_rate'] = float(arg2)
            print('learning_rate=', learning_arguments['learning_rate'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-number_nodes':
            learning_arguments['number_nodes'] = float(arg2)
            print('number_nodes=', learning_arguments['number_nodes'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-max_load_on_server':
            INPUT_DICT['max_load_on_server'] = float(arg2)
            print('number_nodes=', INPUT_DICT['max_load_on_server'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-hidden_units':
            learning_arguments['hidden_units'] = [int(arg2)]
            print('hidden_units=', int(arg2))
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        if arg1 == '-alg':
            INPUT_DICT['alg'] = arg2
            print('Algorithm=', INPUT_DICT['alg'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        
        if arg1 == '-beta':
            INPUT_DICT['beta'] = float(arg2)
            print('beta=',INPUT_DICT['beta'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-v':
            INPUT_DICT['velocity'] = float(arg2)
            print('velocity=',INPUT_DICT['velocity'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        if arg1 == "-federated2":
            INPUT_DICT['federated_learning_param_server'] = float(arg2)
            print('federated_learning_param_server=',INPUT_DICT['federated_learning_param_server'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        if arg1 == '--array_id':
            array=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
            INPUT_DICT['Number of users']=array[int(arg2)]
            print('Number of users=', array[int(arg2)])
            argstr2 = arg_str
            arg_str=arg_str+'nuser'+'_'+str(array[int(arg2)])+'_'
        if arg1 == '-outputpath':
            outputpath = arg2+'results'
            print('outputpath='+arg2)
        
    #INPUT_DICT['Folder']='max_load_on_server_nearest'
    folder = INPUT_DICT['Folder']
    timestr= datetime.now().strftime("%Y_%m_%d")
    outputpath = outputpath+"/"+folder+"_"+timestr
    comment = INPUT_DICT['comment']+'_'
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    Path(outputpath +'/figs').mkdir(parents=True, exist_ok=True)
    Path(outputpath +'/final_results').mkdir(parents=True, exist_ok=True)

    seed = INPUT_DICT['seed']
    print('#############################################')
    for a in INPUT_DICT.keys():
        print(a+' : '+str(INPUT_DICT[a]))
    for a in learning_arguments.keys():
        print(a+' : '+str(learning_arguments[a]))
    print('#############################################')
    for a in GCN_paramaters.keys():
        print(a+' : '+str(GCN_paramaters[a]))
    print('#############################################')

    dp_values ={}
    maximum_of_finish_time={}
    Average_of_finish_time={}
    minimum_of_finish_time={}
    Average_of_optimal_value={}
    Average_of_dp_value ={}
    optimal_values={}
    finish_times={}
    test_Average_of_finish_time={}
    test_minimum_of_finish_time={}
    test_finish_times={}
    Input_dict = INPUT_DICT
    M = Input_dict['Number of users']

    NumberofEpisodes  = INPUT_DICT['Number of episodes']

    arg_str = comment + arg_str
    filename = outputpath  + '/texoutputs/' + arg_str
    filename_png = outputpath  + '/' + 'figs/' + arg_str
    Path(filename).mkdir(parents=True, exist_ok=True)
    Path(filename_png).mkdir(parents=True, exist_ok=True)
    file = open(filename + ".txt", "w+")
    #file_Q = open(filename + "_Q_result.txt", "w+")
    #file_end_results = open(outputpath  + '/final_results/' + arg_str + "_end_result.txt", "w+")
    file_graph_values = open(filename_png + "/values.txt", "w+")
    print('File name: ', filename_png)
    text = 'File name: ' + filename + '\n'
    text += ('#########INPUT_DICT#########' + '\n')
    for a in INPUT_DICT.keys():
        text += (a + ' : ' + str(INPUT_DICT[a]) + '\n')
    text += ('#########learning_arguments#########' + '\n')
    for a in learning_arguments.keys():
        text += (a + ' : ' + str(learning_arguments[a]) + '\n')
    text += ('#############GCN_paramaters##################' + '\n')
    for a in GCN_paramaters.keys():
        text += (a + ' : ' + str(GCN_paramaters[a]) + '\n')
    text += ('#############################################' + '\n')
    
    parameters_text = text
    file.write(text)
    telegram.sendMessage(text)
    telegram.send_running()
    allresult_text=''
    allQ_text=''
    Number_of_runs=INPUT_DICT['Number of runs']
    # set seed for random
    random.seed(seed)
    np.random.seed(seed)
    totaltime=INPUT_DICT['SimulationTime']
    fileiter = open(outputpath+"iterdataoutput.txt", "w+")
    parameters = [10]    
    for param in parameters:
        INPUT_DICT['Number of servers'] = param
        simulator = MEC_Simulator(outputfile=fileiter, Input_dict=Input_dict, learning_arguments=learning_arguments,filename_png=filename_png)

        tau_transmission, tau_waiting, tau_computing = compute_tau_values(T=range(totaltime), tasks = {m:simulator.users[m].tasks for m in range(simulator.M)}, simulator=simulator)
        # Solve using the generated sample dataset
        
        prob,status, objective_value = solve_linearized_problem(M=range(simulator.M), N=range(simulator.S), T=range(totaltime), V_m ={m: simulator.users[m].tasks.keys() for m in range(simulator.M)}, P_m={(m,i):task.predecessors for m in range(simulator.M) for i,task in simulator.users[m].tasks.items()}, S=range(1,simulator.Q+1), K_n=[simulator.input_dict['server capacity']  for _ in range(simulator.S)],i={(m,i):simulator.users[m].tasks[i].input_data_length for m in range(simulator.M) for i in simulator.users[m].tasks.keys()},o={(m,i):simulator.users[m].tasks[i].outputlength for m in range(simulator.M) for i in simulator.users[m].tasks.keys()},z_mi={(m,i):simulator.users[m].tasks[i].service for m in range(simulator.M) for i in simulator.users[m].tasks.keys()},between_server_costs=simulator.between_server_costs, tau_transmission=tau_transmission, 
                                tau_waiting=tau_waiting, tau_computing=tau_computing, l_z=simulator.service_data_length, r_cloud={n:simulator.servers[n].rate_to_cloud for n in range(simulator.S)}, U=100000,lasttasks={m:simulator.users[m].lasttask for m in range(simulator.M)})

        print("Decision Variables:")
        for v in prob.variables():
            print(f"{v.name} = {v.varValue}")
        print(f"Optimal value: {objective_value}")
        print(f"Status: {status}")
        # Save the optimal value in a CSV file

        csv_filename = os.path.join("optimal_values.csv")
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header only if the file does not exist
            if not file_exists:
                writer.writerow(["Status", "Optimal Value"] + list(INPUT_DICT.keys()))
            # Write the data row
            writer.writerow([status, objective_value] + [INPUT_DICT[key] for key in INPUT_DICT.keys()])

        print(f"Optimal value and input parameters appended to {csv_filename}")
        # Print all decision variables

# Output results
    #for m in range(simulator.M):
    #    telegram.send_photo( f"DAG_with_Node_and_Edge_Attributes_user_{simulator.users[m].id}.png",f'DAG_with_Node_and_Edge_Attributes_user_{simulator.users[m].id}')
    
