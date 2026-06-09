import torch
from agent import Agent
from gcn import ServerGCN, TaskGCN
from task import Task
import numpy as np
import tensorflow as tf
import random
import networkx as nx
import time
from input import GCN_paramaters as gcn_paramaters
import pulp

class Broker:
    def __init__(self,env,max_cpu_cycles,max_data_length,numberofservices,numberofservers,learning_arguments,algorithm,filename_png):
        self.simulator=None
        self.env=env
        learning_arguments_broker = learning_arguments.copy()
        learning_arguments_broker['learning_rate'] = 0.01
        self.agent = Agent(algorithm = algorithm,learning_arguments = learning_arguments_broker,numberofservers  = numberofservers,numberofservices = numberofservices,max_cpu_cycles = max_cpu_cycles,max_data_length=max_data_length,filename_png=filename_png)
        self.numberofservices = numberofservices
        self.numberofservers =numberofservers
        #self.alltasks=[]
        self.H = {s: {q: 0 for q in range(1,self.numberofservices+1)} for s in range(self.numberofservers)}

        self.servergcn = ServerGCN(numberofservices=self.numberofservices, hidden_dim=[64, 64], output_dim=10, dropout=0.5)

        self.task_gcn = TaskGCN(self.numberofservices, hidden_dim=gcn_paramaters['layers'], output_dim= gcn_paramaters['output_dim'], dropout=gcn_paramaters['dropout'])
    def make_training_data(self,servers_info):
        inputs=[]
        outputs=[]
        selected_indices = random.sample(range(len(self.all_tasks)), min(1000, len(self.all_tasks)))
        #for index in selected_indices:
        for index in range(len(self.all_tasks)):
            tasks = self.all_tasks[index]
            # TODO GCN
            # DAG_task = self.DAGs[index]
            # node_features = self.task_gcn.generate_task_features(tasks, self.numberofservices)
            # edge_index = self.edgeindexes[index]
            # task_embeddings = self.task_gcn(node_features, edge_index).detach().numpy()
            task_embeddings = None
            for task in tasks.values():
                inputs.append(self.agent.state(task, tasks=tasks, done_tasks=None, deadline=-1, servers_service_info=self.simulator.server_service_info, gat=self.simulator.gat, DAG=task.DAG, Embeddings=task_embeddings))
                output = np.zeros(self.agent.numberofservers)
                for s in servers_info.values():
                    if task.service in s.services:
                        output[s.id] = 1.0
                outputs.append(output)
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        return inputs,outputs
    def make_training_tasks(self,DAGs,DAGsizeMAX,max_cpu_cycles,max_data_length):
        all_tasks=[]
        used_DAGs=[]
        batch_size = 128
        edge_indexes=[]
        for batch_index in range(batch_size):
            for service in range(1,self.numberofservices+1):
                while (True):
                    random_graph_key = random.choice(list(DAGs.keys()))
                    DAG = nx.DiGraph(DAGs[random_graph_key])
                    if(len(DAG.nodes.items())<=DAGsizeMAX):
                        break
                tasks={}
                edge_index=[]
                for edge in DAG.edges:                    
                    if (DAG[edge[0]][edge[1]]['datalength'] >1):
                        print(DAG[edge[0]][edge[1]]['datalength'] )
                    DAG[edge[0]][edge[1]]['datalength'] = int(DAG[edge[0]][edge[1]]['datalength'] *max_data_length)
                    edge_index.append([float(edge[0]), float(edge[1])])
                for i,node in DAG.nodes.items():
                    tasks[i] = Task(user_id = 0,tasknumber=i,cpu_cycles = int(max_cpu_cycles*node['cpucycle']),service=service,DAG=DAG)
                
                all_tasks.append(tasks)
                used_DAGs.append(DAG)
                edge_indexes.append(torch.tensor(edge_index, dtype=torch.long).t().contiguous())
        return all_tasks,used_DAGs,edge_indexes
    def learn0(self,inputs,outputs):
        loss=1000
        while(True):
            loss_t= loss
            with tf.GradientTape() as tape:
                actual_values = self.agent.agent.TrainNet.predict(inputs)
                loss = tf.math.reduce_mean(tf.square(actual_values - outputs))
                variables = self.agent.agent.TrainNet.model.trainable_variables
                gradients = tape.gradient(loss, variables)
            self.agent.agent.TrainNet.optimizer.apply_gradients(zip(gradients, variables))
            #print(loss)
            #print('Diff:')
            #print((loss_t-loss))
            if (np.abs(loss_t-loss)) <1e-2:
                break
    def learn(self, inputs, outputs):
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        # Ensure inputs and outputs are PyTorch tensors and move to device
        #inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        #inputs = torch.stack(list(inputs)).to(dtype=torch.float32, device=device)
        inputs = torch.stack([torch.tensor(i, dtype=torch.float32, device=device) for i in inputs])



        outputs = torch.tensor(outputs, dtype=torch.float32).to(device)

        # Set the model to training mode and move to device
        self.agent.agent.TrainNet.model.train()
        self.agent.agent.TrainNet.model.to(device)

        loss = 1000.0  # Initialize loss
        loss_fn = torch.nn.MSELoss()

        while True:
            loss_t = loss

            # Zero the gradients
            self.agent.agent.TrainNet.optimizer.zero_grad()

            # Forward pass
            actual_values = self.agent.agent.TrainNet.model(inputs)

            # Compute loss
            loss = loss_fn(actual_values, outputs)

            # Backward pass
            loss.backward()

            # Update the weights
            self.agent.agent.TrainNet.optimizer.step()

            # Check for convergence
            #if abs(loss_t - loss.item()) < 1e-5:
            #    break
             # Check for convergence
            if abs(loss.item()) < 1e-2:
                break

    def weights0(self):
        return self.agent.agent.TrainNet.model.get_weights()
    def weights(self):
        return self.agent.agent.TrainNet.model.state_dict()
    def broadcast_weights_user(self,brokerweights):
        beta = self.simulator.beta
        for m in range(self.simulator.M):
            self.simulator.users[m].weights_adjustement(brokerweights,beta)    

    def broadcast_weights_server(self,broker_state_dict):
        coeff = self.simulator.beta
        for server in self.simulator.servers.values():
            local_state_dict = server.agent.agent.TrainNet.model.state_dict()

            # Create a new state_dict to store the averaged weights
            averaged_state_dict = {}

            # Iterate over the parameter names and values
            for name, local_param in local_state_dict.items():
                # Get the corresponding parameter from the broker's model
                broker_param = broker_state_dict[name]

                # Ensure both parameters are on the same device
                broker_param = broker_param.to(local_param.device)

                # Compute the averaged parameter
                averaged_param = (1 - coeff) * local_param + coeff * broker_param

                # Store the averaged parameter in the new state_dict
                averaged_state_dict[name] = averaged_param

            # Load the new averaged state_dict into the local model
            server.agent.agent.TrainNet.model.load_state_dict(averaged_state_dict)
            # Update the target model if necessary
            server.agent.agent.update_target_model()  
    def attach(self,simulator):
        self.simulator=simulator
        self.all_tasks,self.DAGs,self.edgeindexes= self.make_training_tasks(DAGs = self.simulator.loaded_graphs,DAGsizeMAX = self.simulator.I,max_cpu_cycles=self.simulator.max_cpu_cycles,max_data_length=self.simulator.max_data_length)
    def run(self):
        #start_time = time.time()  # Start time for the iteration
        while(self.simulator.notcomplete):
            self.simulator.notcomplete = False
            for m in range(self.simulator.M):
                self.simulator.notcomplete = self.simulator.notcomplete or (not self.simulator.users[m].complete)
            yield self.env.timeout(1)
        #end_time = time.time()  # End time for the iteration
        #elapsed_time = end_time - start_time
        #print(f"broker Iteration took {elapsed_time:.10f} seconds")
        if self.simulator.input_dict['caching decision enabled']:
            self.broadcast_caching_decisions()
        #self.federated_learning()

        
    def caching_decisions_update(self,server,service,cost):
        # Initialization
        alpha = 0.1
        cost = 1
        #self.H[server][service] += alpha * (self.simulator.service_data_length[service] *cost - self.H[server][service])
        #self.H[server][service] +=  (self.simulator.service_data_length[service] *cost)
        #self.H[server][service] += 1
        # Update Averaged Values
        for q in range(1,self.numberofservices+1):
            # Assuming I(q=eta_mi) is an indicator function that returns 1 if q equals eta_mi, else 0
            #self.H[server][q] += alpha * (int(q == service)*self.simulator.service_data_length[q] - self.H[server][q])
            #self.H[server][q] += alpha * (int(q == service)*self.simulator.service_data_length[q] *cost - self.H[server][q])
            self.H[server][q] += alpha * (int(q == service) - self.H[server][q])
            #print(f"Updated H[{server}][{q}] = {self.H[server][q]}")
    def caching_decisions(self,server):
        Ks = self.simulator.input_dict['server capacity']
        # Select Ks services with the highest values in H[q][s]
        sorted_dict = sorted(self.H[server].items(),key=lambda x: x[1], reverse=True)
        top_services = [item[0] for item in sorted_dict[:Ks]]
        return top_services
    
    def broadcast_caching_decisions(self):
        #C=self.caching_decision_solver(N=self.numberofservers,S=self.numberofservices,requestcounter=self.H,servicelength=self.simulator.service_data_length , between_server_costs=self.simulator.between_server_costs,servers_capacity=[self.simulator.input_dict['server capacity']  for _ in range(self.numberofservers)])
        # c={}
        # c[(1,0)] = 0.0
        # c[(1,1)] = 1.0
        # c[(1,2)] = 1.0
        # c[(2,0)] = 0.0
        # c[(2,1)] = 0.0
        # c[(2,2)] = 1.0
        # c[(3,0)] = 1.0
        # c[(3,1)] = 0.0
        # c[(3,2)] = 0.0
        # c[(4,0)] = 1.0
        # c[(4,1)] = 0.0
        # c[(4,2)] = 0.0
        # c[(5,0)] = 0.0
        # c[(5,1)] = 1.0
        # c[(5,2)] = 0.0

        # c[(1,0)] = 0.0
        # c[(1,1)] = 0.0
        # c[(1,2)] = 1.0
        # c[(2,0)] = 1.0
        # c[(2,1)] = 0.0
        # c[(2,2)] = 0.0
        # c[(3,0)] = 1.0
        # c[(3,1)] = 1.0
        # c[(3,2)] = 0.0
        # c[(4,0)] = 0.0
        # c[(4,1)] = 0.0
        # c[(4,2)] = 1.0
        # c[(5,0)] = 0.0
        # c[(5,1)] = 1.0
        # c[(5,2)] = 0.0
        caching_changed = False
        #z={}
        for s in range(self.numberofservers):
            new_decision = self.caching_decisions(s)
            #new_decision = [i+1 for i in range(self.numberofservices) if C[0,s, i] == 1]
            #new_decision = [i+1 for i in range(self.numberofservices) if c[(i+1, s)] == 1]
            if self.simulator.servers[s].services != [0] + new_decision:
                self.simulator.servers[s].services = [0] + new_decision
                caching_changed = True
                for service in range(self.numberofservices):
                    self.simulator.server_service_info[s,service] = 0
                for service in self.simulator.servers[s].services:
                    if service>0:
                        self.simulator.server_service_info[s,service-1] = 1
            #z[s]={} 
            #for q in range(1,self.numberofservices+1):
            #    if q in new_decision:
            #        z[s][q]=1
            #    else:
            #        z[s][q]=0
            #z[s][0]=1
        #TODO THERE IS NO TRANSFER LEARNING....
        #if caching_changed:
            #pass
            #self.simulator.dp.get_service_caching(z)
            #self.simulator.dp.solve()
            #inputs,outputs = self.make_training_data(servers_info = self.simulator.servers)
            #self.learn(inputs,outputs)
            #self.broadcast_weights_server(self.weights())

    def gcn_train(self):
        node_features = self.servergcn.generate_server_features(self.simulator.graph,self.numberofservices)
        edge_index = self.servergcn.generate_edge_index(self.simulator.graph)
        optimizer = torch.optim.Adam(self.servergcn.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.servergcn.train_gcn(node_features,edge_index, optimizer, loss_fn, epochs=100)
        self.servergcn.eval()
        return self.servergcn
    
    def federated_learning(self):
        client_updates = []
        for server in self.simulator.servers.values():  # Assume we have 3 clients
            client_state_dict = server.agent.agent.TrainNet.model.state_dict()  # Dummy state_dict
            coeff = random.random()  # Random coefficient for each client
            client_updates.append((client_state_dict, coeff))
        
        # Perform federated learning
        aggregated_state_dict = self.aggregate(client_updates)
        for server in self.simulator.servers.values():
            server.agent.agent.TrainNet.model.load_state_dict(aggregated_state_dict)

    def aggregate(self, client_updates):
        """
        Perform federated learning by aggregating client updates.

        Parameters
        ----------
        client_updates : list of tuples
            Each tuple contains (client_state_dict, coefficient) for a client.

        Returns
        -------
        None
        """
        # Initialize an empty state_dict to store the aggregated weights
        aggregated_state_dict = {}

        # Iterate over the keys of the first client's state_dict
        for key in client_updates[0][0].keys():
            # Initialize the aggregated parameter with zeros
            aggregated_state_dict[key] = torch.zeros_like(client_updates[0][0][key])

        # Aggregate the client updates
        for client_state_dict, coeff in client_updates:
            for key in client_state_dict.keys():
                aggregated_state_dict[key] += coeff * client_state_dict[key].to(aggregated_state_dict[key].device)

        # Normalize the aggregated weights by the sum of coefficients
        total_coeff = sum([coeff for _, coeff in client_updates])
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] /= total_coeff
        return aggregated_state_dict



    def solve_multi_time_caching_noB(self,T, N, S, K, servicelength, between_server_costs, cap=None):
        """
        Solve the multi-time caching problem without B-step retention:
        Minimize sum_{t,n,s} K[t,n,s] * sum_{n'} ( y[t,n,s,n'] * L[n,n'] ),
        subject to the constraints described above.
        
        Parameters:
        -----------
        T : int - number of time steps
        N : int - number of servers
        S : int - number of services
        K : 3D array-like, shape (T, N, S)
            K[t][n][s] = #requests from server n for service s at time t
        L : 2D array-like, shape (N, N)
            L[n][n'] = cost to fetch from n' to n
        cap : list of length N or None
            cap[n] = max #services that server n can hold at once, or None if unlimited
        
        Returns:
        --------
        status, C_sol, Y_sol, min_cost
        """
        prob = pulp.LpProblem("MultiTimeNoB", pulp.LpMinimize)
        # Decision vars: C[t,n,s], Y[t,n,s,n']
        C = {}
        Y = {}
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    C[(t,n,s)] = pulp.LpVariable(f"C_{t}_{n}_{s}", cat=pulp.LpBinary)
                    for nprime in range(N):
                        Y[(t,n,s,nprime)] = pulp.LpVariable(f"Y_{t}_{n}_{s}_{nprime}", cat=pulp.LpBinary)

        # Objective: sum_t,n,s [ K[t,n,s] * sum_{n'} ( Y[t,n,s,n'] * L[n][n'] ) ]
        prob += pulp.lpSum(K[n][s+1] * Y[(t,n,s,nprime)] * servicelength[s+1] * between_server_costs[n][nprime] for t in range(T) for n in range(N) for s in range(S) for nprime in range(N)), "TotalFetchCost"

        # Constraint (A): sum_{n'} Y[t,n,s,n'] = 1 - C[t,n,s]
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    prob += pulp.lpSum(Y[(t,n,s,nprime)] for nprime in range(N)) == (1 - C[(t,n,s)])

        # Constraint (B): Y[t,n,s,n'] <= C[t,n',s]
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    for nprime in range(N):
                        prob += Y[(t,n,s,nprime)] <= C[(t,nprime,s)]

        # Constraint (C): capacity (optional)
        if cap is not None:
            for t in range(T):
                for n in range(N):
                    prob += pulp.lpSum(C[(t,n,s)] for s in range(S)) <= cap[n]

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        status = pulp.LpStatus[prob.status]
        min_cost = pulp.value(prob.objective)
        
        # Extract solutions
        C_sol = { (t,n,s): int(pulp.value(C[(t,n,s)])) for t in range(T) for n in range(N) for s in range(S) }
        Y_sol = { (t,n,s,nprime): int(pulp.value(Y[(t,n,s,nprime)])) for t in range(T) for n in range(N) for s in range(S) for nprime in range(N) }

        return status, C_sol, Y_sol, min_cost

    def caching_decision_solver(self,N,S,requestcounter,servicelength, between_server_costs,servers_capacity):
        T=1
        status, C_sol, Y_sol, cost_val = self.solve_multi_time_caching_noB(T, N, S, K = requestcounter, servicelength= servicelength, between_server_costs = between_server_costs, cap=servers_capacity)
        #print("Solver Status:", status)
        #print("Minimized Cost:", cost_val)
        #for t in range(T):
        #    for n in range(N):
        #        cached = [s for s in range(S) if C_sol[(t,n,s)]==1]
        #        print(f"Time={t}, Server={n}, Cached Services:", cached)
        return C_sol
# Example usage
if __name__ == "__main__":
    # T=2, N=2, S=2
    T, N, S = 2, 2, 2
    # K[t][n][s]
    K = [
        [[10, 5],  [3, 8]],  # t=0
        [[2,  4],  [6, 1]]   # t=1
    ]
    # L[n][n']
    L = [
        [0, 2],
        [2, 0]
    ]
    # capacity?
    cap = [1, 1]  # each server can cache 1 service at a time

 
