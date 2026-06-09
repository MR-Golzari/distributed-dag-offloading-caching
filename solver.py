import gurobipy as gp
import os
from gurobipy import GRB
from task import Task
from server import Server
import numpy as np
class Optimizer:
    def __init__(self,num_v: int,num_s : int,tasks):
        self.c={}
        self.service={}
        self.P={}
        self.len_out={}
        self.len_in={}
        self.V = []
        self.tasks = tasks
        self.solved = False
        for t in tasks.values():
            self.V.append(int(t.task_number))
            self.c[int(t.task_number)] = t.cpu_cycle
            self.service[int(t.task_number)] = t.service
            self.P[int(t.task_number)]=[]
            self.len_in[int(t.task_number)]=t.input_data_length
            for tp in t.predecessors:
                self.P[int(t.task_number)].append(int(tp))
                self.len_out[int(t.task_number),int(self.tasks[tp].task_number)] = self.tasks[tp].outputs_length[t.task_number]
            if t.successors ==[]:
                self.last_task = int(t.task_number)
    def get_server_and_service_parameters(self,servers,service_lengths,server_rates,server_latencies,to_servers_rate,nearest_server):
        self.num_s = len(servers)
        self.num_service = len(service_lengths)
        self.tau_load={}
        self.f={}
        self.tau_w={}
        #self.z={}
        self.server_rates=server_rates
        self.to_servers_rate = to_servers_rate
        self.server_latencies = server_latencies
        self.nearest_server= nearest_server
        print("nearest_server: ",nearest_server)
        for s in servers.values():
            self.f[s.id] = s.frequency
            self.tau_w[s.id] = s.load*(10**6)/s.frequency 
            #self.z[s.id]=np.zeros((self.num_service))
                   
            for i in range(self.num_service):
                self.tau_load[s.id,i] = service_lengths[i]/s.rate_to_cloud
            
            #for serv in s.services:
                #self.z[s.id][serv] = 1

    def solve_minlp(self):
        try:   
            # Create a new Gurobi model
            self.model = gp.Model("MINLP_problem")
            self.model.setParam("Threads", int(os.getenv("GRB_THREADS", "16")))

            # Decision variables x[i, s]
            self.x = {}  # Dictionary to store the decision variables
            self.T_pred = {}
            for i in self.V:
                for s in range(self.num_s):
                    self.x[i, s] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")
                    self.T_pred[i,s] = self.model.addVar(name=f"T_pred_{i}_{s}")

            # Define the service variables
            self.z={}
            for s in range(self.num_s):
                self.z[s]={}
                for service in range(self.num_service):
                    self.z[s][service] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{service}")
            # Define the computation time T_i
            self.T= {}
            for i in self.V:
                self.T[i] = self.model.addVar(name=f"T_{i}")
            # Add transmission time constraint
            self.delta={}
            for i in self.V:
                for j in self.V:
                    for s in range(self.num_s):
                        self.delta[i,j,s] =self.model.addVar(name=f"delta_{i}_{j}_{s}")
                        if j in self.P[i]:
                            self.model.addConstr(self.delta[i,j, s] == sum(self.x[j, s_prime] * self.tau_f(s,s_prime,self.len_out[i,j]) for s_prime in range(self.num_s)))
                        else:
                            self.model.addConstr(self.delta[i,j, s] == 0)

            # Add the service variables constraint
            for s in range(1,self.num_s):
                self.model.addConstr(sum(self.z[s][service] for service in range(1,self.num_service))==1)
                self.model.addConstr(self.z[s][0] ==1)
            
            # Define the objective function T_m|V|
            self.application_finish_time=self.model.addVar(name=f"application finish time")
            self.model.setObjective(self.application_finish_time, GRB.MINIMIZE)

            # Add constraints
            self.model.addConstr(self.application_finish_time==self.T[self.last_task]+sum(self.x[self.last_task, s]*self.server_latencies[s,self.nearest_server] for s in range(self.num_s)))
            for i in self.V:
                self.model.addConstr(sum(self.x[i, s] for s in range(self.num_s)) == 1, name=f"constraint_{i}")

            # Add computation time constraint
            for i in self.V:
                self.model.addConstr(self.T[i] == sum(self.x[i, s] * (self.tau_data(self.len_in[i],s)+self.c[i] / self.f[s] + (1 - self.z[s][self.service[i]]) * self.tau_load[s,self.service[i]] + self.tau_w[s] + self.T_pred[i,s]) for s in range(self.num_s)))
                for s in range(self.num_s):
                     for j in self.P[i]:
                        self.model.addConstr(self.T_pred[i,s] >=self.T[j] + self.delta[i,j,s])
            # Solve the model
            self.model.optimize()

            if self.model.status == GRB.OPTIMAL:
                # Get the optimal solution
                self.optimal_x = {}
                for i in self.V:
                    for s in range(self.num_s):
                        self.optimal_x[i, s] = self.model.getVarByName(f"x_{i}_{s}").x
                self.optimal_z = {}
                for s in range(self.num_s):
                    self.optimal_z[s] = {}
                    for service in range(self.num_service):
                        self.optimal_z[s][service] = self.model.getVarByName(f"z_{s}_{service}").x

                self.optimal_objective = self.model.objVal
                self.solved = True
        except gp.GurobiError as e:
            print("Error code", e.errno, ":", e)
        
    def print_solution(self):
        if self.solved:
            print("Optimal Solution:")
            print("Objective Value:", self.optimal_objective)
            for i in self.V:
                for s in range(self.num_s):
                    print(f"x_{i}_{s} =", self.optimal_x[i, s])
        else:
            print("No optimal solution found.")

    def tau_f(self,s,sp,l):
        if s==sp:
            return 0
        else:
            return l/self.server_rates[sp,s]+self.server_latencies[sp,s]
    
    def tau_data(self,l,s):
        if l==0:
            return 0
        else:
            return l/self.to_servers_rate[s]+self.server_latencies[self.nearest_server,s]
        
        
    def printvalues(self):
        name = "application finish time"
        print(name+":",self.model.getVarByName(name).x)
        for i in self.V:
            name = f"T_{i}"
            print(name+":",self.model.getVarByName(name).x)

        for i in self.V:
            for s in range(self.num_s):
                name = f"x_{i}_{s}"
                print(name+":",self.model.getVarByName(name).x)

 
        for i in self.V:
            for s in range(self.num_s):
                name = f"T_pred_{i}_{s}"
                print(name+":",self.model.getVarByName(name).x)
                #for j in self.V:
                #    name = f"delta_{i}_{j}_{s}"
                #    print(name+":",self.model.getVarByName(name).x)
        
        for s in range(self.num_s):
            for service in range(self.num_service):
                name = f"z_{s}_{service}"
                print(name+":",self.model.getVarByName(name).x) 



class Joint_Optimizer:
    def __init__(self,M,num_s : int,tasks):
        self.M = M
        self.c={}
        self.service={}
        self.P={}
        self.len_out={}
        self.len_in={}
        self.V = {}
        self.last_task={}
        self.tasks = tasks
        self.solved = False
        self.optimal_objective = 0 
        self.optimal_finishtime = {}
        for m in range(self.M):
            self.optimal_finishtime[m] = 0
        for m in range(self.M):
            self.V[m] = []
            for t in tasks[m].values():
                self.V[m].append(int(t.task_number))
                self.c[m,int(t.task_number)] = t.cpu_cycle
                self.service[m,int(t.task_number)] = t.service
                self.P[m,int(t.task_number)]=[]
                self.len_in[m,int(t.task_number)]=t.input_data_length
                for tp in t.predecessors:
                    self.P[m,int(t.task_number)].append(int(tp))
                    self.len_out[m,int(t.task_number),int(self.tasks[m][tp].task_number)] = self.tasks[m][tp].outputs_length[t.task_number]
                if t.successors ==[]:
                    self.last_task[m] = int(t.task_number)
    def get_server_and_service_parameters(self,servers,service_lengths,server_rates,server_latencies,to_servers_rate,nearest_server):
        self.num_s = len(servers)
        self.num_service = len(service_lengths)
        self.tau_load={}
        self.f={}
        self.tau_w={}
        #self.z={}
        self.server_rates=server_rates
        self.to_servers_rate = to_servers_rate
        self.server_latencies = server_latencies
        self.nearest_server= nearest_server
        print("nearest_server: ",nearest_server)
        for s in servers.values():
            self.f[s.id] = s.frequency
            self.tau_w[s.id] = s.load*(10**6)/s.frequency 
            #self.z[s.id]=np.zeros((self.num_service))
                   
            for i in range(self.num_service):
                self.tau_load[s.id,i] = service_lengths[i]/s.rate_to_cloud
            
            #for serv in s.services:
                #self.z[s.id][serv] = 1

    def solve_minlp(self):
        try:   
            # Create a new Gurobi model
            self.model = gp.Model("MINLP_problem")
            self.model.setParam("Threads", int(os.getenv("GRB_THREADS", "16")))
            self.model.setParam("TimeLimit", 120) 

            # Decision variables x[i, s]
            self.x={}
            self.T_pred = {}
            for m in range(self.M):
                for i in self.V[m]:
                    for s in range(self.num_s):
                        self.x[m,i, s] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{m}_{i}_{s}")
                        self.T_pred[m,i,s] = self.model.addVar(name=f"T_pred_{m}_{i}_{s}")

            # Define the service variables
            self.z={}
            for s in range(self.num_s):
                for service in range(self.num_service):
                    self.z[s,service] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{service}")
            
            # Define the computation time T_m_i
            self.T= {}
            self.delta={}
            for m in range(self.M):

                for i in self.V[m]:
                    self.T[m,i] = self.model.addVar(name=f"T_{m}_{i}")
                # Add transmission time constraint
                for i in self.V[m]:
                    for j in self.V[m]:
                        for s in range(self.num_s):
                            self.delta[m,i,j,s] =self.model.addVar(name=f"delta_{m}_{i}_{j}_{s}")
                            if j in self.P[m,i]:
                                self.model.addConstr(self.delta[m,i,j, s] == sum(self.x[m,j, s_prime] * self.tau_f(s,s_prime,self.len_out[m,i,j]) for s_prime in range(self.num_s)))
                            else:
                                self.model.addConstr(self.delta[m,i,j, s] == 0)

            # Add the service variables constraint
            for s in range(self.num_s):
                self.model.addConstr(sum(self.z[s,service] for service in range(1,self.num_service))==1)
                self.model.addConstr(self.z[s,0] ==1)
            
            
            # Define the objective function T_m|V|
            self.application_finish_time={}
            for m in range(self.M):
                self.application_finish_time[m]=self.model.addVar(name=f"application finish time_{m}")

                # Add constraints
                self.model.addConstr(self.application_finish_time[m]==self.T[m,self.last_task[m]]+sum(self.x[m,self.last_task[m], s]*self.server_latencies[s,self.nearest_server[m]] for s in range(self.num_s)))
                for i in self.V[m]:
                    self.model.addConstr(sum(self.x[m,i, s] for s in range(self.num_s)) == 1, name=f"constraint_{m}_{i}")
                # Add computation time constraint
                    self.model.addConstr(self.T[m,i] == sum(self.x[m,i, s] * (self.tau_data(self.len_in[m,i],s,m)+self.c[m,i] / self.f[s] + (1 - self.z[s,self.service[m,i]]) * self.tau_load[s,self.service[m,i]] + self.tau_w[s] + self.T_pred[m,i,s]) for s in range(self.num_s)))
                    for s in range(self.num_s):
                        for j in self.P[m,i]:
                            self.model.addConstr(self.T_pred[m,i,s] >=self.T[m,j] + self.delta[m,i,j,s])
            
            self.model.setObjective(np.mean([self.application_finish_time[m] for m in range(self.M)]), GRB.MINIMIZE)
            # Solve the model
            self.model.optimize()
            if True:
            #if self.model.status == GRB.OPTIMAL :
                # Get the optimal solution
                self.optimal_x = {}
                for m in range(self.M):
                    for i in self.V[m]:
                        for s in range(self.num_s):
                            self.optimal_x[m,i, s] = self.model.getVarByName(f"x_{m}_{i}_{s}").x
                self.optimal_z = {}
                for s in range(self.num_s):
                    self.optimal_z[s] = {}
                    for service in range(self.num_service):
                        self.optimal_z[s][service] = self.model.getVarByName(f"z_{s}_{service}").x

                self.optimal_finishtime={}
                for m in range(self.M):
                    self.optimal_finishtime[m]=self.model.getVarByName(f"application finish time_{m}").x

                self.optimal_objective = self.model.objVal
                self.solved = True
        except gp.GurobiError as e:
            print("Error code", e.errno, ":", e)
        
    def print_solution(self):
        if self.solved:
            print("Optimal Solution:")
            print("Objective Value:", self.optimal_objective)
            for m in range(self.M):
                for i in self.V[m]:
                    for s in range(self.num_s):
                        print(f"x_{m}_{i}_{s} =", self.optimal_x[m,i, s])
            for s in range(self.num_s):
                for service in range(self.num_service):
                        print(f"z_{s}_{service}=", self.optimal_z[s][service])
            
            print("Objective Value:", self.optimal_objective)
        else:
            print("No optimal solution found.")

    def tau_f(self,s,sp,l):
        if s==sp:
            return 0
        else:
            return l/self.server_rates[sp,s]+self.server_latencies[sp,s]
    
    def tau_data(self,l,s,m):
        if l==0:
            return 0
        else:
            return l/self.to_servers_rate[m][s]+self.server_latencies[self.nearest_server[m],s]
        
        
    def printvalues(self):
        for m in range(self.M):
            name = f"application finish time_{m}"
            print(name+":",self.model.getVarByName(name).x)
            for i in self.V[m]:
                name = f"T_{m}_{i}"
                print(name+":",self.model.getVarByName(name).x)

            for i in self.V[m]:
                for s in range(self.num_s):
                    name = f"x_{m}_{i}_{s}"
                    print(name+":",self.model.getVarByName(name).x)

 
            for i in self.V[m]:
                for s in range(self.num_s):
                    name = f"T_pred_{m}_{i}_{s}"
                    print(name+":",self.model.getVarByName(name).x)
                    #for j in self.V:
                    #    name = f"delta_{i}_{j}_{s}"
                    #    print(name+":",self.model.getVarByName(name).x)
        
        for s in range(self.num_s):
            for service in range(self.num_service):
                name = f"z_{s}_{service}"
                print(name+":",self.model.getVarByName(name).x) 

class DynamicProgramming:
    def __init__(self,M,num_s : int,tasks):
        self.M = M
        self.c={}
        self.service={}
        self.P={}
        self.len_out={}
        self.len_in={}
        self.V = {}
        self.last_task={}
        self.tasks = tasks
        self.solved = False
        for m in range(self.M):
            self.V[m] = []
            for t in tasks[m].values():
                self.V[m].append(int(t.task_number))
                self.c[m,int(t.task_number)] = t.cpu_cycle
                self.service[m,int(t.task_number)] = t.service
                self.P[m,int(t.task_number)]=[]
                self.len_in[m,int(t.task_number)]=t.input_data_length
                for tp in t.predecessors:
                    self.P[m,int(t.task_number)].append(int(tp))
                    self.len_out[m,int(t.task_number),int(self.tasks[m][tp].task_number)] = self.tasks[m][tp].outputs_length[t.task_number]
                if t.successors ==[]:
                    self.last_task[m] = int(t.task_number)
    def get_server_and_service_parameters(self,servers,service_lengths,server_rates,server_latencies,to_servers_rate,nearest_server):
        self.num_s = len(servers)
        self.num_service = len(service_lengths)
        self.tau_load={}
        self.f={}
        self.tau_w={}
        #self.z={}
        self.server_rates=server_rates
        self.to_servers_rate = to_servers_rate
        self.server_latencies = server_latencies
        self.nearest_server= nearest_server
        print("nearest_server: ",nearest_server)
        for s in servers.values():
            self.f[s.id] = s.frequency
            self.tau_w[s.id] = s.load*(10**6)/s.frequency 
            #self.z[s.id]=np.zeros((self.num_service))
                   
            for i in range(self.num_service):
                self.tau_load[s.id,i] = service_lengths[i]/s.rate_to_cloud
            
            #for serv in s.services:
                #self.z[s.id][serv] = 1
    def get_service_caching(self,z_input):
        self.z = z_input.copy()
    def solve(self):
        # Initialize DP table, prev_time, and z variables
        DP = {}
        prev_time = {}
        backtrack = {}
        self.delta ={}
        for m in range(self.M):
            DP[m] = {}
            prev_time[m] = {}
            for i in self.V[m]:
                DP[m][i] = {}
                prev_time[m][i] = 0
                for s in range(self.num_s):
                    DP[m][i][s] = float('inf')


        # Fill DP table
        for m in range(self.M):
            for i in self.V[m]:
                backtrack[m,i]={}
                for s in range(self.num_s):

                    if len(self.P[m,i]) > 0:
                        for j in self.P[m,i]:
                            ex_prev_time_s_prime = float('inf')
                            for s_prime in range(self.num_s):
                                prev_time_s_prime = DP[m][j][s_prime] + self.tau_f(s,s_prime,self.len_out[m,i,j])
                                if prev_time_s_prime <= ex_prev_time_s_prime:
                                    ex_prev_time_s_prime = prev_time_s_prime
                                    backtrack[m,i][j] = s_prime
                                
                            prev_time[m][i] = max(prev_time[m][i],ex_prev_time_s_prime)
                        DP[m][i][s] = (prev_time[m][i] + 
                                   self.tau_data(self.len_in[m,i], s, m) + 
                                   self.c[m,i] / self.f[s] + 
                                   (1 - self.z[s][self.service[m,i]]) * self.tau_load[s,self.service[m,i]] + 
                                   self.tau_w[s])
                    else:
                        
                        DP[m][i][s] = (self.tau_data(self.len_in[m,i], s, m) + 
                                       self.c[m,i] / self.f[s] + 
                                       (1 - self.z[s][self.service[m,i]]) * self.tau_load[s,self.service[m,i]] + 
                                       self.tau_w[s])
                        backtrack[m,i] = -1  # Indicates a starting task


        # Compute application finish time and determine the optimal assignment
        self.application_finish_time = {}
        self.optimal_assignment = {}

        for m in range(self.M):
            self.optimal_assignment[m] = {}
            min_finish_time = float('inf')
            best_server = -1
            for s in range(self.num_s):
                finish_time = DP[m][self.last_task[m]][s] + self.server_latencies[s, self.nearest_server[m]]
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_server = s
            self.optimal_assignment[m][self.last_task[m]] = best_server

            # Backtrack to find the optimal assignment for all tasks
            for i in reversed(self.V[m]):
                for j in self.P[m,i]:
                    if j not in self.optimal_assignment[m]:
                        self.optimal_assignment[m][j] = backtrack[m,i][j]
        finish_time={}
        for m in range(self.M):
            for i in self.V[m]:
                    finish_time[m,i] = 0
                    s = self.optimal_assignment[m][i] 
                    if len(self.P[m,i]) > 0:
                        ex_prev_time = float('inf')
                        prev_time = max([finish_time[m,j]+ self.tau_f(s,self.optimal_assignment[m][j] ,self.len_out[m,i,j]) for j in self.P[m,i]])
                        finish_time[m,i] = (prev_time + 
                                   self.tau_data(self.len_in[m,i], s, m) + 
                                   self.c[m,i] / self.f[s] + 
                                   (1 - self.z[s][self.service[m,i]]) * self.tau_load[s,self.service[m,i]] + 
                                   self.tau_w[s])
                    else:
                        
                        finish_time[m,i] = (self.tau_data(self.len_in[m,i], s, m) + 
                                       self.c[m,i] / self.f[s] + 
                                       (1 - self.z[s][self.service[m,i]]) * self.tau_load[s,self.service[m,i]] + 
                                       self.tau_w[s])


        # Objective: minimize average application finish time
            self.application_finish_time[m] = finish_time[m,self.last_task[m]]
        self.objective = np.mean([self.application_finish_time[m] for m in range(self.M)])

        self.solved= True

        return self.objective, self.optimal_assignment
        
    def print_solution(self):
        if self.solved:
            # Print the results
            print("Objective:", self.objective)
            print("Optimal Assignment of Tasks to Servers:")
            for m in range(self.M):
                print(f"Application {m} finish time: {self.application_finish_time[m]}")
                for i in self.V[m]:
                    print(f"Application {m}, Task {i}: Server {self.optimal_assignment[m][i]}")
        else:
            print("No optimal solution found.")

    def tau_f(self,s,sp,l):
        if s==sp:
            return 0
        else:
            return l/self.server_rates[sp,s]+self.server_latencies[sp,s]
    
    def tau_data(self,l,s,m):
        if l==0:
            return 0
        else:
            return l/self.to_servers_rate[m][s]+self.server_latencies[self.nearest_server[m],s]
        
        