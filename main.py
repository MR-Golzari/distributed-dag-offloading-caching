# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
import random
from user import User
from server import Server
from utils import plot_users_servers
from input import INPUT_DICT
from input import learning_arg as learning_arguments
from input import GCN_paramaters
from simulator import MEC_Simulator
from operator import attrgetter
from datetime import datetime
import sys
from tqdm import tqdm
import os
from pathlib import Path

from telegram import Telegram
import json,time
import pulp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import random

def compute_tau_values(tasks, servers, simulator):
    tau_transmission = {}
    tau_waiting = {}
    tau_computing = {}
    tau_service = {}

    for m, task_list in tasks.items():
        for i, task in task_list.item():
            for n, server in servers:

                # Transmission latency (assumed pre-calculated)
                tau_transmission[m, i, n] =  task.input_data_length / simulator.user[m].to_servers_rate[n]

                # Waiting latency (based on server load)
                waiting_latency = server.load * (1e6) / server.frequency
                tau_waiting[m, i, n] = waiting_latency

                # Computing latency (CPU cycles required / server frequency)
                computing_latency = task.cpu_cycle / server.frequency
                tau_computing[m, i, n] = computing_latency

                # Determine service loading conditions
                if task.service in server.services:
                    serviceloadingcost = 0
                else:
                    if len(server.servers_with_service) == 0:
                        serviceloadingcost = 1.0 / server.rate_to_cloud
                    else:
                        serviceloadingcost = min(
                            [simulator.between_server_costs[other_server.id, n] 
                             for other_server in server.servers_with_service]
                        )

                # Service latency calculation
                service_latency = simulator.service_data_length[task.service] * serviceloadingcost
                tau_service[m, i, n] = service_latency


    return tau_transmission, tau_waiting, tau_computing, tau_service

def moving_averages_calc(values):
    window_size =200
    k = 0
    moving_averages = {}
    arr = list(values)
    for k in range(len(arr)):
         # Calculate the average of current window
        if (k<window_size):
            window_average = np.sum(arr[0:k+1])/(k+1)
        else:
            window_average = np.sum(arr[k-window_size:k]) / window_size
        moving_averages[k]=window_average
    return moving_averages
def extractoutput(key,moving_averages,outputpath,arg_str):
    # Define the path where the JSON file will be saved
    folder_path = outputpath
    file_name = arg_str+"_numberofusers.json"
    file_path = os.path.join(folder_path, file_name)

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Calculate the average of the last 100 values
    last_100_values = list(moving_averages.values())[-100:]
    average_last_100 = sum(last_100_values) / len(last_100_values) if last_100_values else 0

    new_data = {key: average_last_100}

    # Read the existing data, update it, and write it back
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)  # Load existing data
    else:
        data = {}  # Initialize an empty dictionary if the file doesn't exist

    # Update the data with the new entry
    data.update(new_data)

    # Write the updated data back to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

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
            print('max_load_on_server=', INPUT_DICT['max_load_on_server'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
            
        if arg1 == '-max_rate_between_servers':
            INPUT_DICT['max_rate_between_servers'] = float(arg2)
            print('max_rate_between_servers=', INPUT_DICT['max_rate_between_servers'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-max_cpu_frquency':
            INPUT_DICT['max_cpu_frquency'] = float(arg2)
            print('max_cpu_frquency=', INPUT_DICT['max_cpu_frquency'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-max_cpu_cycles':
            INPUT_DICT['max_cpu_cycles'] = float(arg2)
            print('max_cpu_cycles=', INPUT_DICT['max_cpu_cycles'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-hidden_units':
            learning_arguments['hidden_units'] = [int(arg2)]
            print('hidden_units=', int(arg2))
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        if arg1 == '-alg':
            INPUT_DICT['alg'] = arg2
            print('Algorithm=', INPUT_DICT['alg'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        if arg1 == '-enable_caching':
            INPUT_DICT['caching decision enabled'] = arg2.lower() in ('true', '1', 'yes')
            print('caching decision enabled=', INPUT_DICT['caching decision enabled'])
            arg_str=arg_str+arg1[1:]+'_'+'cached'+'_'+str(INPUT_DICT['caching decision enabled'])+'_'
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
            array=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
            INPUT_DICT['seed']=array[int(arg2)]
            print('seed=', array[int(arg2)])
            argstr2 = arg_str
            arg_str=arg_str+'seed'+'_'+str(array[int(arg2)])+'_'
        if arg1 == '-outputpath':
            outputpath = arg2+'results'
            print('outputpath='+arg2)
        
    #INPUT_DICT['Folder']='max_load_on_server_nearest'
    folder = INPUT_DICT['Folder']
    timestr= datetime.now().strftime("%Y_%m_%d")
    outputpath = outputpath+"/"+timestr+"_"+folder
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
    fileiter = open(outputpath+"/iterdataoutput.txt", "w+")
    simulator = MEC_Simulator(outputfile=fileiter, Input_dict=Input_dict, learning_arguments=learning_arguments,filename_png=filename_png)
    text2=''

    computing_share_over_time = {j: [] for j in range(Number_of_runs)}
    data_transfer_share_over_time = {j: [] for j in range(Number_of_runs)}
    pred_share_over_time = {j: [] for j in range(Number_of_runs)}
    service_share_over_time = {j: [] for j in range(Number_of_runs)}
    waiting_share_over_time = {j: [] for j in range(Number_of_runs)}

    simulator.plot_positions()
    telegram.send_photo('locations.png','Initial Locations')
    telegram.send_photo('graph_of_servers.png','Graph of Servers')
    telegram.send_photo('mst_of_servers.png','spanning tree of Graph of Servers')
    
    #tau_transmission, tau_waiting, tau_computing, tau_service = compute_tau_values(tasks = {m:simulator.users[m].tasks for m in range(simulator.M)}, servers = simulator.servers, simulator=simulator)
    #solve_linearized_problem(M=range(simulator.M), N=range(simulator.S), T=range(1), V_m={m: simulator.user[m].numberoftasks for m in range(simulator.M)}, P_m={(m,i):simulator.user[m].tasks[i].predecessors for m in range(simulator.M) for i in range(simulator.user[m].numberoftasks)}, S=simulator.Q, K_n=[simulator.input_dict['server capacity']  for _ in range(simulator.S)], T_pred=simulator.T_pred, tau_f=simulator.tau_f, tau_transmission=simulator.tau_transmission, tau_waiting=simulator.tau_waiting, tau_computing=simulator.tau_computing, l_z=simulator.l_z, r_n=simulator.r_n, z_mi=simulator.z_mi)

    #for m in range(simulator.M):
    #    telegram.send_photo( f"DAG_with_Node_and_Edge_Attributes_user_{simulator.users[m].id}.png",f'DAG_with_Node_and_Edge_Attributes_user_{simulator.users[m].id}')
    for j in range(Number_of_runs):
        print(f'Run: {j + 1}/{Number_of_runs}')
        maximum_of_finish_time[j]={}
        Average_of_finish_time[j]={}
        minimum_of_finish_time[j]={}
        Average_of_optimal_value[j]={}
        Average_of_dp_value[j]={}
        optimal_values[j]={}
        dp_values[j]={}
        finish_times[j]={}
        Input_dict['seed'] = random.randint(0,1000)
        for m in range(simulator.M):    
            simulator.users[m].setpos0()
                #for iter in range(NumberofEpisodes):
        for iter in tqdm(range(NumberofEpisodes), desc="Processing", unit="iteration", ncols=100):
            #start_time = time.time()  # Start time for the iteration
            #print('')
            #print('Progress: %',100*iter/NumberofEpisodes)
            telegram.Progress(iter,NumberofEpisodes)
            simulator.reset()
            optimal_values[j][iter],finish_times[j][iter],dp_values[j][iter]=simulator.run()
            Average_of_finish_time[j][iter] = np.mean(finish_times[j][iter])
            Average_of_optimal_value[j][iter] = np.mean(optimal_values[j][iter])
            Average_of_dp_value[j][iter] = np.mean(dp_values[j][iter])
            minimum_of_finish_time[j][iter] = np.min(optimal_values[j][iter])
            maximum_of_finish_time[j][iter] = np.max(optimal_values[j][iter])
            file_graph_values.write(str(Average_of_finish_time[j][iter])+'\n')



            # Store the contribution shares for this iteration
            computing_share_over_time[j].append(simulator.computing_share)
            data_transfer_share_over_time[j].append(simulator.data_transfer_share)
            pred_share_over_time[j].append(simulator.pred_share)
            service_share_over_time[j].append(simulator.service_share)
            waiting_share_over_time[j].append(simulator.waiting_share)

            #simulator.plot_positions(iter)
            #telegram.send_photo('locations.png',str(j)+" "+str(iter)+": average finishtime: "+str(Average_of_finish_time[j][iter]))
            #end_time = time.time()  # End time for the iteration
            #elapsed_time = end_time - start_time
            #print(f"main Iteration took {elapsed_time:.10f} seconds")
       # for m in range(M):
       #     simulator.users[m].agent.brain.save_model()
        #simulator.reset()
        #result_text,Q_text,test_Average_of_finish_time[j],test_minimum_of_finish_time[j],test_finish_times[j] = simulator.test(j)
        #allresult_text+=result_text
        #allQ_text+=Q_text
        #text2+='Average of finish epoc: '+ str(test_Average_of_finish_time[j])+'\n'
    #text = text2
    #text+='Average of finish time_allaverage: '+ str(np.average(list(test_Average_of_finish_time.values())))+'\n'
    #text+='minimum of finish time_allaverage: '+ str(np.average(list(test_minimum_of_finish_time.values())))+'\n'
    #file.write('#####################+\n')
    #file_Q.write(text+'\n###############\n'+allQ_text+'\n###############\n'+allresult_text)
    #file_end_results.write(text+'\n###############\n'+allresult_text+'\n###############\n'+text+'\n###############\n'+allQ_text+'\n###############\n')
    #file.write(allQ_text+'\n')
    #file.write(allresult_text)
    #Average_of_finish_time[j] = np.average(list(Average_of_finish_time[j].values()))
    #minimum_of_finish_time[j] = np.average(list(minimum_of_finish_time[j].values()))
    file.close()
    #file_Q.close()
    #file_end_results.close()
    file_graph_values.close()

    Average_of_finish_time_average={}
    Average_of_optimal_value_average={}
    Average_of_dp_value_average={}

    for ne in range(NumberofEpisodes):
        Average_of_finish_time_average[ne] = np.mean([Average_of_finish_time[j][ne] for j in range(Number_of_runs)])
        Average_of_optimal_value_average[ne] = np.mean([Average_of_optimal_value[j][ne] for j in range(Number_of_runs)])
        Average_of_dp_value_average[ne] = np.mean([Average_of_dp_value[j][ne] for j in range(Number_of_runs)])

    plt.figure()
    plt.plot(range(NumberofEpisodes), Average_of_finish_time_average.values(), '-*',label='Algorithm')
    #plt.plot(range(NumberofEpisodes), Average_of_optimal_value_average.values(), '--',label='Optimal')
    #plt.plot(range(NumberofEpisodes), Average_of_dp_value_average.values(), '--',label='DP')

    plt.xlabel('Iteration')
    plt.ylabel('Average of finish time')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename_png+'/average.pdf',dpi=300)
    #plt.show()
    for m in range(simulator.M):
        Average_of_finish_time_average_user={}
        Average_of_optimal_value_average_user={}
        Average_of_dp_value_average_user ={}
        for ne in range(NumberofEpisodes):
            Average_of_finish_time_average_user[ne] = np.mean([finish_times[j][ne][m] for j in range(Number_of_runs)])
            Average_of_optimal_value_average_user[ne] = np.mean([optimal_values[j][ne][m] for j in range(Number_of_runs)])
            Average_of_dp_value_average_user[ne] = np.mean([dp_values[j][ne][m] for j in range(Number_of_runs)])

        moving_averages = moving_averages_calc(Average_of_finish_time_average_user.values())
        plt.figure()
        plt.plot(range(NumberofEpisodes), moving_averages.values(), '-',label='Algorithm')
        #plt.plot(range(NumberofEpisodes), Average_of_optimal_value_average_user.values(), '--',label='Optimal')
        #plt.plot(range(NumberofEpisodes), Average_of_dp_value_average_user.values(), '--',label='DP')

        plt.xlabel('Iteration')
        plt.ylabel('Average of finish time')
        plt.grid(True)
        plt.title('Application Finish Time Convergence Over Time')

        #plt.legend()
        #plt.savefig(filename_png+f'/average_{m}.pdf',dpi=300)

    minimum_of_finish_time_average={}
    for ne in range(NumberofEpisodes):
        minimum_of_finish_time_average[ne] = np.mean([minimum_of_finish_time[j][ne] for j in range(Number_of_runs)])
    plt.figure()
    plt.plot(range(NumberofEpisodes), minimum_of_finish_time_average.values(), '-*')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum of finish time')
    plt.title('Application Finish Time Convergence Over Time')
    plt.grid()
    #plt.legend()
    plt.savefig(filename_png+'/minumum.png')
    #plt.show()

    moving_averages=moving_averages_calc(Average_of_finish_time_average.values())
    

    plt.figure()
    plt.plot(range(NumberofEpisodes), moving_averages.values(), '-',label='Algorithm')
    #plt.plot(range(NumberofEpisodes), Average_of_optimal_value_average.values(), '--',label='Optimal')
    #plt.plot(range(NumberofEpisodes), Average_of_dp_value_average.values(), '--',label='DP')

    plt.xlabel('Iteration')
    plt.ylabel('Average of finish time')
    plt.title('Application Finish Time Convergence Over Time')

    plt.grid()
    #plt.legend()
    plt.savefig(filename_png+'/moving_averages.pdf',dpi=300)
    print(filename_png)
        # After all runs are complete, calculate the average contribution shares across runs for each iteration
    avg_computing_share_over_time = np.mean([computing_share_over_time[j] for j in range(Number_of_runs)], axis=0)
    avg_data_transfer_share_over_time = np.mean([data_transfer_share_over_time[j] for j in range(Number_of_runs)], axis=0)
    avg_pred_share_over_time = np.mean([pred_share_over_time[j] for j in range(Number_of_runs)], axis=0)
    avg_service_share_over_time = np.mean([service_share_over_time[j] for j in range(Number_of_runs)], axis=0)
    avg_waiting_share_over_time = np.mean([waiting_share_over_time[j] for j in range(Number_of_runs)], axis=0)
    # Plot the average contribution shares over iterations
    iterations = range(NumberofEpisodes)

    moving_averages_avg_computing_share_over_time = moving_averages_calc(avg_computing_share_over_time)
    moving_averages_avg_data_transfer_share_over_time = moving_averages_calc(avg_data_transfer_share_over_time)
    moving_averages_avg_service_share_over_time = moving_averages_calc(avg_service_share_over_time)
    moving_averages_avg_waiting_share_over_time = moving_averages_calc(avg_waiting_share_over_time)
    plt.figure()
    # Write the average contribution shares to a file
    shares_file_path = os.path.join(filename_png, 'latency_shares.txt')
    with open(shares_file_path, 'w') as shares_file:
        shares_file.write("Iteration,Total,Computing Share,Data Transfer Share,Service Share,Waiting Share\n")
        for i in iterations:
            shares_file.write(f"{i},{moving_averages[i]:.6f},"
                              f"{moving_averages_avg_computing_share_over_time[i]:.6f},"
                              f"{moving_averages_avg_data_transfer_share_over_time[i]:.6f},"
                              f"{moving_averages_avg_service_share_over_time[i]:.6f},"
                              f"{moving_averages_avg_waiting_share_over_time[i]:.6f}\n")
    print(f"Latency contribution shares written to {shares_file_path}")
    plt.plot(iterations, moving_averages.values(), '-',label='Algorithm')
    plt.plot(iterations, moving_averages_avg_computing_share_over_time.values(), label='Computing Latency Share')
    plt.plot(iterations, moving_averages_avg_data_transfer_share_over_time.values(), label='Data Transfer Latency Share')
    #plt.plot(iterations, avg_pred_share_over_time, label='Predessesor Latency Share')
    plt.plot(iterations, moving_averages_avg_service_share_over_time.values(), label='Service Latency Share')
    plt.plot(iterations, moving_averages_avg_waiting_share_over_time.values(), label='Waiting Latency Share')
    plt.ylim(0, 0.180)
    plt.xlabel('Iterations')
    plt.ylabel('Average Contribution Share')
    plt.title('Average Latency Contribution Share Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_png+'/shares.pdf',dpi=300)
    plt.savefig(filename_png+'/shares.png',dpi=300)
    telegram.send_photo(filename_png+'/shares.png','shares')
    #plt.show()
    #try:
    #telegram.sendresults(parameters_text,text,simulator.M,optimal_values,test_finish_times,Number_of_runs,filename_png,result_text,filename)

    #plt.figure()
    #plt.plot(M_list,maximum_of_finish_time.values(),'-*')
    #plt.xlabel('Number of Users')
    #plt.ylabel('Maximum of finish time')
    #plt.grid()

    #plt.show()

    #plt.figure()
    #plt.plot(M_list,Average_of_finish_time.values(),'-*')
    #plt.xlabel('Number of Users')
    #plt.ylabel('Average of finish time')
    #plt.grid()

    #plt.show()

    #plt.figure()
    #plt.plot(M_list,minimum_of_finish_time.values(),'-*')
    #plt.xlabel('Number of Users')
    #plt.ylabel('Minimum of finish time')
    #plt.grid()

    #plt.show()

    extractoutput(arg_str,moving_averages,outputpath,argstr2)
    print('Finished')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

