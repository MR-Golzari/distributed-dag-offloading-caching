# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
#from ortools.linear_solver import pywraplp
import numpy as np
import random
from user import User
from server import Server
from utils import plot_users_servers
from input import INPUT_DICT
from operator import attrgetter
import json
import networkx as nx
from solver import Joint_Optimizer,DynamicProgramming
from agent import Agent
from networkx.drawing.nx_pydot import graphviz_layout
from broker import Broker
import simpy 
import matplotlib.pyplot as plt
import numpy as np
import random
from user import User
from server import Server
from utils import plot_users_servers
from input import INPUT_DICT
from input import learning_arg as learning_arguments

from simulator import MEC_Simulator
from operator import attrgetter
from datetime import datetime
import sys
from tqdm import tqdm
import os
from pathlib import Path
from telegram import Telegram
class optimization:   
    def __init__(self,outputfile,Input_dict,learning_arguments,filename_png):
        self.env = simpy.Environment()
        self.notcomplete = True
        # Read parameters from input
        self.filename_png=filename_png
        self.file = outputfile
        self.alg =Input_dict['alg']
        self.M = Input_dict['Number of users']
        self.S = Input_dict['Number of servers']
        self.I = Input_dict['Number of tasks for each user']
        self.Q = Input_dict['Number of services']
        #self.Ks = INPUT_DICT['Ks']
        self.xlim=Input_dict['xlim']
        self.ylim=Input_dict['ylim']
        self.max_cpu_cycles=Input_dict['max_cpu_cycles']
        self.max_data_length=Input_dict['max_data_length']
        self.SimulationTime= Input_dict['SimulationTime']
        self.power = Input_dict['Power']
        self.BW = Input_dict['Bandwidth']
        self.min_cpu_freq = Input_dict['min_cpu_frquency']
        self.max_cpu_freq = Input_dict['max_cpu_frquency']
        self.min_service_data_length = 1e6*Input_dict['min_service_data_length']
        self.max_service_data_length = 1e6*Input_dict['max_service_data_length']
        self.minload = Input_dict['min_load_on_server']
        self.maxload = Input_dict['max_load_on_server']

        self.min_rate_between_servers = Input_dict['min_rate_between_servers']
        self.max_rate_between_servers = Input_dict['max_rate_between_servers']
        self.filling_steps = Input_dict['filling steps']
        self.steps_b_updates = Input_dict['steps to updates']
        self.deadline = Input_dict['deadline']
        self.updatedeadline = Input_dict['update deadline']

        self.maxratetocloud=Input_dict['Max rate to cloud']
        self.minratetocloud=Input_dict['Min rate to cloud']
        #self.agent = Agent(algorithm = self.alg,learning_arguments = learning_arguments,numberofservers  = self.S,numberofservices = self.Q,max_cpu_cycles = self.max_cpu_cycles,max_data_length=self.max_data_length )
        self.tau_t_mis = {}
        self.tau_w_mis = {}
        self.tau_c_mis = {}
        self.time_step = 0
        self.users={}
        self.servers={}
        self.learning_arguments = learning_arguments
        self.broker = Broker(self.env,max_cpu_cycles=self.max_cpu_cycles,max_data_length=self.max_data_length, numberofservices=self.Q, numberofservers=self.S,learning_arguments=learning_arguments,algorithm=self.alg,filename_png=self.filename_png)
        
        with open('dag_uniform.json', 'r') as file:
            loaded_data = json.load(file)
            self.loaded_graphs = {}
            for graph_name, graph_data in loaded_data.items():
                self.loaded_graphs[graph_name] = nx.node_link_graph(graph_data)
        for m in range(self.M):
            while (True):
                random_graph_key = random.choice(list(self.loaded_graphs.keys()))
                random_graph = self.loaded_graphs[random_graph_key]
                if(len(random_graph.nodes.items())<=self.I):
                    break
            #random_graph_key = list(loaded_graphs.keys())[m]
            #random_graph_key = 'j_13027'
            #TODO PYDOT ERROR
            #pos = graphviz_layout(random_graph, prog='dot')
            #plt.figure()
            #nx.draw_networkx(random_graph, pos, with_labels=True, node_color='lightblue')
            #plt.savefig(self.filename_png+f'/DAG_{m}.png')
            # initialzie users
            acc = (0,0)
            v0 = (0,0)
            self.users[m] = User(self.env,id=m, xlim=self.xlim,ylim=self.ylim,  application_graph =random_graph, max_cpu_cycles=self.max_cpu_cycles,
                         max_data_length=self.max_data_length, numberofservices=self.Q, acc=acc, v0=v0, power=self.power,
                         bandwidth=self.BW, numberofservers=self.S,deadline=self.deadline,learning_arguments=learning_arguments,algorithm=self.alg,filename_png=self.filename_png)
        self.service_data_length={}
        self.service_data_length[0] = 0
        for q in range(self.Q):
            self.service_data_length[q+1] = self.min_service_data_length+random.random()*(self.max_service_data_length-self.min_service_data_length)
        
        self.server_latency=np.zeros((self.S,self.S))
        for s in range(0,self.S):
            for sp in range(s+1,self.S):
                    self.server_latency[s,sp]= 1e-3*random.randint(1,5)/10
                    self.server_latency[sp,s]= self.server_latency[s,sp]
        #self.servers[0] = Server(self.env,id=0, numberofservices=self.Q, min_freq=self.min_cpu_freq, max_freq=self.max_cpu_freq,
                                 #xlim=self.xlim, ylim=self.ylim, iscloud=True,minload=self.maxload-1,maxload=self.maxload)
        for s in range(self.S):
            # initialzie servers
            self.servers[s] = Server(self.env,id=s, numberofservices=self.Q, min_freq=self.min_cpu_freq,max_freq=self.max_cpu_freq,xlim=self.xlim, ylim=self.ylim,iscloud=False,minload=self.minload,maxload=self.maxload,minratetocloud=self.minratetocloud,maxratetocloud=self.maxratetocloud)
        
        for m in range(self.M):
            self.users[m].attach(self)
        for n in range(self.S):
            self.servers[n].attach(self)
        self.broker.attach(self)

        self.server_rates=np.zeros((self.S,self.S))
        for s in range(self.S):
            for sp in range(self.S):
                if s==sp:
                    self.server_rates[s,sp] = np.inf
                else:
                    self.server_rates[s,sp]= 1e6*random.randint(self.min_rate_between_servers,self.max_rate_between_servers)



        for m in range(self.M):
            #self.users[m].set_agent(self.agent)
            self.users[m].assign_nearest_server(self.servers)

        self.optimizer = Joint_Optimizer(M=self.M,num_s=self.S,tasks= [self.users[m].tasks_init for m in range(self.M)])
        self.optimizer.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = [self.users[m].to_servers_rate for m in range(self.M)],nearest_server = [self.users[m].nearest_server for m in range(self.M)]) 
        self.optimizer.solve_minlp()
        #self.optimizer.printvalues()
        #self.optimizer.print_solution()


        self.dp = DynamicProgramming(M=self.M,num_s=self.S,tasks= [self.users[m].tasks_init for m in range(self.M)])
        self.dp.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = [self.users[m].to_servers_rate for m in range(self.M)],nearest_server = [self.users[m].nearest_server for m in range(self.M)]) 
        self.dp.get_service_caching(self.optimizer.optimal_z)
        self.dp.solve()
        #self.dp.print_solution()
if __name__ == '__main__':
    telegram = Telegram()
    arg_str=""
    args = sys.argv[1:]
    file_id=""
    while(len(args)):
        arg1=args.pop(0)
        arg2=args.pop(0)
        if arg1 == '-comment':
            INPUT_DICT['comment'] = arg2
            print('folder=',INPUT_DICT['comment'])
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
            title = 'Number of servers'
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'

        if arg1 == '-nuser':
            INPUT_DICT['Number of users'] = int(arg2)
            print('Number of users=',INPUT_DICT['Number of users'])
            arg_str=arg_str+arg1[1:]+'_'+arg2+'_'
        
        if arg1 == '-nservice':
            INPUT_DICT['Number of services'] = int(arg2)
            print('Number of services=',INPUT_DICT['Number of services'])
            title='Number of services'
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
            title='max_service_data_length'
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
        
        if arg1 == '-array_id':
            array=[2,4,8,12,16,20,24,28]
            INPUT_DICT['Number of servers']=array[int(arg2)]
            print('Number of servers=', array[int(arg2)])
            arg_str=arg_str+arg1[1:]+'_'+str(array[int(arg2)])+'_'
    #INPUT_DICT['Folder']='max_load_on_server_nearest'
    folder = INPUT_DICT['Folder']+'_'+INPUT_DICT['alg']
    
    comment = INPUT_DICT['comment']+'_'

    seed = INPUT_DICT['seed']
    print('#############################################')
    for a in INPUT_DICT.keys():
        print(a+' : '+str(INPUT_DICT[a]))
    for a in learning_arguments.keys():
        print(a+' : '+str(learning_arguments[a]))
    print('#############################################')


    Input_dict = INPUT_DICT
    M = Input_dict['Number of users']

    NumberofEpisodes  = INPUT_DICT['Number of episodes']

    arg_str = folder+comment + arg_str
    timestr= datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    filename_png = 'results/' + timestr + '/' + 'figs/' + arg_str

    # set seed for random
    random.seed(seed)
    np.random.seed(seed)
    gorubi=[]
    dp=[]
    file_write = open(f"results/optimization/output_{title}_{INPUT_DICT[title]}.txt", "w+")
    for iter in range(20):
        opt = optimization(outputfile=file_write, Input_dict=Input_dict, learning_arguments=learning_arguments,filename_png=filename_png)
        gorubi.append(opt.optimizer.optimal_objective)
        dp.append(opt.dp.objective)
    file_write = open(f"results/optimization/output_{title}_{INPUT_DICT[title]}.txt", "w+")
    file_write.write(f"{title}: {INPUT_DICT[title]}\n")
    file_write.write(f"gurobi: {np.mean(gorubi)}\n")
    file_write.write(f"dynamic programming: {np.mean(dp)}")
    file_write.close()
