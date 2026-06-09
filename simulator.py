# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
#from ortools.linear_solver import pywraplp
import numpy as np
import random

import torch
from user import User
from server import Server
from utils import plot_users_servers,generate_streets_with_users
from input import INPUT_DICT
from operator import attrgetter
import json
import networkx as nx
from solver import Joint_Optimizer,DynamicProgramming
from agent import Agent
from networkx.drawing.nx_pydot import graphviz_layout
from broker import Broker
from gat import GAT
import simpy 
import time
from gcn import ServerGCN

def shannon_capacity(d):
    B = 50e9  # Bandwidth in Hz (50 GHz)
    P_trans = 1e-3  # Transmitted power in W (1 mW)
    alpha_db_per_km = 0.2  # Attenuation in dB/km
    alpha = (alpha_db_per_km / 10) * np.log(10)  # Convert dB/km to linear
    N0 = 1e-20  # Noise power spectral density in W/Hz
    
    return B * np.log2(1 + (P_trans * np.exp(-alpha * d)) / (N0 * B))

class MEC_Simulator:
    def __init__(self,outputfile,Input_dict,learning_arguments,filename_png):
        self.env = simpy.Environment()
        self.networkgraph = nx.Graph()  # Create an empty graph
        self.input_dict = Input_dict
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
        self.beta = Input_dict['beta']

        self.velocity = Input_dict['velocity']
        self.learning_arguments=learning_arguments
        #self.agent = Agent(algorithm = self.alg,learning_arguments = learning_arguments,numberofservers  = self.S,numberofservices = self.Q,max_cpu_cycles = self.max_cpu_cycles,max_data_length=self.max_data_length )
        self.tau_t_mis = {}
        self.tau_w_mis = {}
        self.tau_c_mis = {}
        self.time_step = 0
        self.users={}
        self.servers={}
        
        self.broker = Broker(self.env,max_cpu_cycles=self.max_cpu_cycles,max_data_length=self.max_data_length, numberofservices=self.Q, numberofservers=self.S,learning_arguments=learning_arguments,algorithm=self.alg,filename_png=self.filename_png)
        
        Numberofstreets_x = 5
        Numberofstreets_y = 5
        self.street_width = 24
        self.min_spacing = (self.ylim-Numberofstreets_x*24)/(Numberofstreets_x-1)
        self.max_spacing = self.min_spacing
        street_positions_y, street_positions_x, user_positions, user_directions = generate_streets_with_users(Numberofstreets_x,Numberofstreets_y, self.M, self.street_width, self.xlim, self.ylim, self.min_spacing, self.max_spacing)
        self.street_positions_y = street_positions_y
        self.street_positions_x = street_positions_x
        self.user_positions = user_positions
        self.user_directions = user_directions
        self.vmin=self.velocity/10
        self.vmax = self.velocity 
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
            acc = (0.0,0.0)
            v0 = random.randint(-100,100)/100000
            self.users[m] = User(self.env,id=m, pos0 =user_positions[m] ,  application_graph =random_graph, max_cpu_cycles=self.max_cpu_cycles,
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
            server = Server(self.env,id=s, numberofservices=self.Q, min_freq=self.min_cpu_freq,max_freq=self.max_cpu_freq,xlim=self.xlim, ylim=self.ylim,iscloud=False,minload=self.minload,maxload=self.maxload,minratetocloud=self.minratetocloud,maxratetocloud=self.maxratetocloud)
            self.servers[s] = server
            self.networkgraph.add_node(s, server=server)  # Add server as a node to the graph

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
                    if np.linalg.norm(np.array(self.servers[s].pos) - np.array(self.servers[sp].pos)) < 5000:
                        #self.server_rates[s,sp]= shannon_capacity(np.linalg.norm(np.array(self.servers[s].pos) - np.array(self.servers[sp].pos)))
                        self.server_rates[s,sp]= 1e9*(self.max_rate_between_servers-(self.max_rate_between_servers-self.min_rate_between_servers)*np.linalg.norm(np.array(self.servers[s].pos) - np.array(self.servers[sp].pos))/5000)
                        self.networkgraph.add_edge(s, sp, weight=1.0/self.server_rates[s,sp])  # Add edge with rate parameter to represent connection between servers
        # Create a minimum spanning tree (MST) based on the networkgraph
        mst = nx.minimum_spanning_tree(self.networkgraph, weight='weight')
        self.networkgraph = mst
        # Draw the MST
        pos = {server.id: server.pos for server in self.servers.values()}  # Use server positions for the graph layout
        plt.figure()
        nx.draw(mst, pos, with_labels=True, node_size=50, node_color='green', font_size=8, font_weight='bold')
        plt.title('Minimum Spanning Tree of Servers')
        plt.savefig('mst_of_servers.png')
        plt.close()
        self.between_server_costs = np.zeros((self.S, self.S))
        for s in range(self.S):
            for sp in range(self.S):
                # Find the path between two nodes in the MST
                source_node = s  # Replace with your source node
                target_node = sp  # Replace with your target node
                path = nx.shortest_path(self.networkgraph, source=source_node, target=target_node, weight='weight')
                # Calculate the sum of weights along the path
                total_weight = sum(mst[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                self.between_server_costs[s, sp] = total_weight
        for m in range(self.M):
            #self.users[m].set_agent(self.agent)
            self.users[m].nearest_server =self.users[m].find_nearest_server(self.servers)
        
        #self.gat = GAT(self.users)
        self.gat = None

        #self.optimizer = Joint_Optimizer(M=self.M,num_s=self.S,tasks= [self.users[m].tasks_init for m in range(self.M)])
        #self.optimizer.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = [self.users[m].to_servers_rate for m in range(self.M)],nearest_server = [self.users[m].nearest_server for m in range(self.M)]) 
        #self.optimizer.solve_minlp()
        #self.optimizer.printvalues()
        
        #self.dp = DynamicProgramming(M=self.M,num_s=self.S,tasks= [self.users[m].tasks_init for m in range(self.M)])
        #self.dp.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = [self.users[m].to_servers_rate for m in range(self.M)],nearest_server = [self.users[m].nearest_server for m in range(self.M)]) 
        self.server_service_info = np.zeros((self.S,self.Q))
        for s in self.servers.values():
            for service in s.services:
                if service>0:
                    self.server_service_info[s.id,service-1] = 1

        if caching_decision := Input_dict.get('caching decision enabled', True):    
            self.broker.broadcast_caching_decisions()
        
        """
        
        for m in range(self.M):
           services={}
           for n in range(self.users[m].numberofservers):
               services[n] = [key for key, value in self.optimizer.optimal_z[int(n)].items() if value >= 0.9]
           pos={}
           listofnodes=list(self.users[m].DAG.nodes())
           for n in listofnodes:
               if n=='0':
                   pos[n]=(10*self.users[m].nearest_server,0)
               else:
                   y = -10*list(nx.topological_sort(self.users[m].DAG)).index(n)
                   x= 10*np.argmax([self.optimizer.optimal_x[m,int(n),s] for s in range(self.users[m].numberofservers)])
                   pos[n]=(x,y)
           plt.figure(3)
           nx.draw_networkx(self.users[m].DAG, pos, with_labels=True, node_color='lightblue')
           plt.grid(True)
           for t in self.users[m].tasks.values():
               plt.text(pos[t.task_number][0]+5, pos[t.task_number][1]+5, t.service, ha='center', fontsize=10, color='red')

            # Create a graph
           servers_graph = nx.Graph()
           pos_s={}
           for n in range(self.S):
                   servers_graph.add_node(n, service = services[n])
                   pos_s[n]=(n *10, 10)

                # Draw the graph with the grid layout
           nx.draw_networkx(servers_graph, pos=pos_s, with_labels=True, node_size=300, node_color='red', font_size=10)
           for n in range(self.S):
               plt.text(pos_s[n][0], pos_s[n][1]+5, services[n][1], ha='center', fontsize=10, color='red')

           plt.savefig(self.filename_png+f'/DAG_sol{m}.png')
        #for n in range(self.users[m].numberofservers):
        #    self.servers[n].service_caching(services[n])
        """    
    def reset(self):
        for m in range(self.M):
            # initialzie users
            acc = (0,0)
            v0 = (0,0)
            #self.select_random_graph(m,acc,v0)
            #acc = (10 * (random.random() - 0.5), 10 * (random.random() - 0.5))
            #v0 = (10 * (random.random() - 0.5), 10 * (random.random() - 0.5))
            self.users[m].reset()

    def select_random_graph_and_find_optimal_offloading_decision(self,m,acc,v0):
            
            while (True):
                random_graph_key = random.choice(list(self.loaded_graphs.keys()))
                random_graph = self.loaded_graphs[random_graph_key]
                if(len(random_graph.nodes.items())<=self.I):
                    break
            pos = graphviz_layout(random_graph, prog='dot')
            plt.figure()
            nx.draw_networkx(random_graph, pos, with_labels=True, node_color='lightblue')
            plt.savefig(self.filename_png+f'/DAG_{m}/{random_graph_key}.png')
            numberoftasks = self.users[m].task_generate(DAG = random_graph,
                             max_cpu_cycles=self.max_cpu_cycles,
                             max_data_length=self.max_data_length, numberofservices=self.Q, acc=acc, v0=v0,
                             power=self.power,
                             bandwidth=self.BW, numberofservers=self.S)
            
            self.users[m].optimizer = Optimizer(num_v =numberoftasks,num_s=self.S,tasks= self.users[m].tasks_init)
            self.users[m].optimizer.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = self.users[m].to_servers_rate,nearest_server = self.users[m].nearest_server)
            self.users[m].optimizer.solve_minlp()
            #self.users[m].optimizer.printvalues()
    
            pos={}
            listofnodes=list(self.users[m].DAG.nodes())
            for n in listofnodes:
                if n=='0':
                    pos[n]=(10*self.users[m].nearest_server,0)
                else:
                    y = -10*list(nx.topological_sort(self.users[m].DAG)).index(n)
                    x= 10*np.argmax([self.users[m].optimizer.optimal_x[int(n),s] for s in range(self.users[m].numberofservers)])
                    pos[n]=(x,y)
            plt.figure(2)
            nx.draw_networkx(self.users[m].DAG, pos, with_labels=True, node_color='lightblue')
            plt.grid(True)
            for t in self.users[m].tasks.values():
                plt.text(pos[t.task_number][0]+5, pos[t.task_number][1]+5, t.service, ha='center', fontsize=10, color='red')

            # Create a graph
            servers_graph = nx.Graph()
            pos_s={}
            for n in range(self.users[m].numberofservers):
                    servers_graph.add_node(n, service = self.servers[n].services)
                    self.servers[n].services
                    pos_s[n]=(n *10, 10)

                # Draw the graph with the grid layout
            nx.draw_networkx(servers_graph, pos=pos_s, with_labels=True, node_size=300, node_color='red', font_size=10)
            for n in range(self.users[m].numberofservers):
                plt.text(pos_s[n][0], pos_s[n][1]+5, self.servers[n].services[1], ha='center', fontsize=10, color='red')

            plt.savefig(self.filename_png+f'/DAG_{m}/{random_graph_key}_sol.png')
    def test(self,epoc):
        text='~~~~~~~\n#run'+str(epoc)+'\n'
        Q_text='~~~~~~~\n#run'+str(epoc)+'\n'
        observations, tasks = self.get_tasks_and_observations(self.time_step)
        done_list={}
        for m in range(self.M):
            done_list[m] = False
            self.users[m].agent.epsilon=0
        complete = False
        Q = {}
        self.time_step=0

        while (not complete):
            actions = self.select_actions(observations,tasks)
            for m in range(self.M):
                Q[m] = self.users[m].agent.agent.TrainNet.predict(np.atleast_2d(observations[m]))[0]
            Q_text += 'Time: '+str(self.time_step) + '\n Q: '+str(Q)+'\n'
            rewards,done_list,complete = self.step(self.time_step,actions,tasks,done_list)
            next_observations, tasks = self.get_tasks_and_observations(self.time_step)
            Q_text +=str(observations)+'\n'+str(actions)+'\n'+str(next_observations)+'\n'+str(rewards)+'\n#########################\n'
            observations = next_observations
            self.time_step += 1
        text+='Deadline: '+str([self.users[m].deadline for m in range(self.M)])+'\n'
        Average_of_finish_time = np.mean([self.users[m].finish_time_of_application for m in range(self.M)])
        minimum_of_finish_time = min([self.users[m].finish_time_of_application for m in range(self.M)])
        text+='Average of finish time: '+ str(Average_of_finish_time)+'\n'
        text+='minimum of finish time: '+ str(minimum_of_finish_time)+'\n'
        finish_times = [self.users[m].finish_time_of_application for m in range(self.M)]
        
        for m in range(self.M):
            pos = {}
            listofnodes=list(self.users[m].DAG.nodes())
            for n in listofnodes:
                if n=='0':
                    pos[n]=(10*self.users[m].nearest_server,0)
                else:
                    y = -10*list(nx.topological_sort(self.users[m].DAG)).index(n)
                    x= 10*self.users[m].done_tasks[n].assigned_server
                    pos[n]=(x,y)
            plt.figure(4)
            nx.draw_networkx(self.users[m].DAG, pos, with_labels=True, node_color='lightblue')
            plt.grid(True)
            for t in self.users[m].tasks.values():
                plt.text(pos[t.task_number][0]+5, pos[t.task_number][1]+5, t.service, ha='center', fontsize=10, color='red')
            # Create a graph
            servers_graph = nx.Graph()
            pos_s={}
            for n in range(self.users[m].numberofservers):
                    servers_graph.add_node(n, service = self.servers[n].services)
                    self.servers[n].services
                    pos_s[n]=(n *10, 10)
                # Draw the graph with the grid layout
            nx.draw_networkx(servers_graph, pos=pos_s, with_labels=True, node_size=300, node_color='red', font_size=10)
            for n in range(1, self.users[m].numberofservers):
                plt.text(pos_s[n][0], pos_s[n][1]+5, self.servers[n].services[1], ha='center', fontsize=10, color='red')
            plt.savefig(self.filename_png+f'/DAG_{m}_RL.png')    
        return text,Q_text,Average_of_finish_time,minimum_of_finish_time,finish_times
    def run(self):
        #start_time = time.time()  # Start time for the iteration
        self.notcomplete=True
        for m in range(self.M):
            self.env.process(self.users[m].run())
        for n in range(self.S):
            self.env.process(self.servers[n].run())
        self.env.process(self.broker.run())
        self.env.run()
        Simulation_Time = self.time_step
        Average_of_finish_time = np.mean([self.users[m].finish_time_of_application for m in range(self.M)])
        minimum_of_finish_time = min([self.users[m].finish_time_of_application for m in range(self.M)])
        #average_optimal_value = self.optimizer.optimal_objective
        #optimal_values = [self.optimizer.optimal_finishtime[m] for m in range(self.M)]
        optimal_values = [self.users[m].finish_time_of_application for m in range(self.M)]
        #dp_values = [self.dp.application_finish_time[m] for m in range(self.M)]
        dp_values = [self.users[m].finish_time_of_application for m in range(self.M)]


        finish_times = [self.users[m].finish_time_of_application for m in range(self.M)]
        

        #print('Simulation Time: ', Simulation_Time)
        #print('Average of finish time: ', Average_of_finish_time)
        #print('minimum of finish time: ', minimum_of_finish_time)
        self.calculate_latency_shares()
        #end_time = time.time()  # End time for the iteration
        #elapsed_time = end_time - start_time
        #print(f"simulator Iteration took {elapsed_time:.10f} seconds")

        return optimal_values,finish_times,dp_values

    def plot_positions(self,iter=0):
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 8))
        server_positions = []
        user_positions = []

        # Plot horizontal streets with spacing
        for y in self.street_positions_y:
            ax.plot([0, self.xlim], [y, y], color='green', linewidth=1, linestyle='--', label='Horizontal Street' if y == self.street_positions_y[0] else "")
            # Show the width of each street
            ax.fill_between([0, self.xlim], y, y + self.street_width, color='green', alpha=0.2)

        # Plot vertical streets with spacing
        for x in self.street_positions_x:
            ax.plot([x, x], [0, self.ylim], color='purple', linewidth=1, linestyle='--', label='Vertical Street' if x == self.street_positions_x[0] else "")
            # Show the width of each street
            ax.fill_betweenx([0, self.ylim], x, x + self.street_width, color='purple', alpha=0.2)

        # Collect server and user positions
        for s in self.servers.values():
            server_positions.append(s.pos)
        for u in self.users.values():
            user_positions.append(u.pos)

        # Plot servers (red circles) and users (blue squares)
        if server_positions:
            ax.scatter(*zip(*server_positions), color='red', marker='o', label='Servers')
        if user_positions:
            ax.scatter(*zip(*user_positions), color='blue', marker='s', label='Users')

        # Set plot limits, labels, and title
        ax.set_xlim(0, self.xlim)
        ax.set_ylim(0, self.ylim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Server and User Positions')
        ax.grid(True)

        # Add legend
        ax.legend()
        fig.savefig(f'locations.png')

        pos = {server.id: server.pos for server in self.servers.values()}  # Use server positions for the graph layout
        nx.draw(self.networkgraph, pos, with_labels=True, ax=ax, node_size=5, node_color='red', font_size=1, font_weight='bold')
        #fig.savefig('graph_of_servers.png')
        #plt.close(fig)
                # Save the figure as 'locations.png'
        # Create a minimum spanning tree (MST) based on the server distances
        #mst = nx.minimum_spanning_tree(self.networkgraph, weight='weight')
        #nx.draw(mst, pos, with_labels=True, ax=ax, node_size=5, node_color='red', font_size=1, font_weight='bold')
        # Save the MST figure as 'mst_of_servers.png'
        fig.savefig(self.filename_png+'/graph_of_servers.pdf', dpi=300)  # Save as PDF with tight bounding box

        plt.close(fig)

        # plt.figure()
        # plt.title("Graph of Servers")
        # plt.savefig('graph_of_servers.png')  # Save the graph as an image file
        # plt.close()
        #plt.show()  


    def calculate_latency_shares(self):
        # Collect latency components from all users
        computing_latencies = [self.users[m].sum_computinglatency for m in range(self.M)]
        data_transfer_latencies = [self.users[m].sum_datatransferlatency for m in range(self.M)]
        pred_latencies = [self.users[m].sum_predlatency for m in range(self.M)]
        service_latencies = [self.users[m].sum_servicelatency for m in range(self.M)]
        waiting_latencies = [self.users[m].sum_waiting_latency for m in range(self.M)]

        # Compute the average latency components across all users
        self.avg_computing_latency = np.mean(computing_latencies)
        self.avg_data_transfer_latency = np.mean(data_transfer_latencies)
        self.avg_pred_latency = np.mean(pred_latencies)
        self.avg_service_latency = np.mean(service_latencies)
        self.avg_waiting_latency = np.mean(waiting_latencies)

        # Calculate the total latency for each user and the average of total latencies
        total_latencies = [comp + dt + pred + serv + wait for comp, dt, pred, serv, wait in zip(
            computing_latencies, data_transfer_latencies, 
            pred_latencies, service_latencies, 
            waiting_latencies)]
        avg_total_latency = np.mean(total_latencies)
        avg_total_latency = 1
        # Calculate the contribution share for each latency component
        self.computing_share = self.avg_computing_latency / avg_total_latency
        self.data_transfer_share = self.avg_data_transfer_latency / avg_total_latency
        self.pred_share = self.avg_pred_latency / avg_total_latency
        self.service_share = self.avg_service_latency / avg_total_latency
        self.waiting_share = self.avg_waiting_latency / avg_total_latency

        # Optional: print or store the values for debugging
        #print(f"Avg Computing Latency Share: {self.computing_share}")
        #print(f"Avg Data Transfer Latency Share: {self.data_transfer_share}")
        #print(f"Avg Prediction Latency Share: {self.pred_share}")
        #print(f"Avg Service Latency Share: {self.service_share}")
        #print(f"Avg Waiting Latency Share: {self.waiting_share}")