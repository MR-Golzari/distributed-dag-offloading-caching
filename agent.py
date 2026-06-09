from dqn import DQN_Agent,GCN_DQN_Agent
from a2c import A2C_Agent
from lstm import LSTM_Agent
from task import Task
from tensorflow.keras.utils import plot_model
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch
import torch.nn as nn
from input import GCN_paramaters as gcn_paramaters

class Agent:
    def __init__(self,algorithm,learning_arguments,numberofservers,numberofservices, max_cpu_cycles,max_data_length,filename_png):
        self.alg = algorithm
        self.numberofservers = numberofservers
        self.numberofservices=numberofservices
        self.action_size = numberofservers
        self. max_cpu_cycles = max_cpu_cycles
        self.max_data_length = max_data_length
        if self.alg == 'simpleDQN':
            self.state_size = 2
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'prev_servers_plus_service_per_serverDQN':
            self.state_size  = numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'prev_serversDQN':
            self.state_size  = numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'simpleA2C':   
            self.state_size = 2
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'prev_serversA2C':
            self.state_size  = numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'justserviceDQN':
            self.state_size  = 2*numberofservices+2+1  #current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'justserviceA2C':
            self.state_size  = 2*numberofservices+2+1  #current services+ next services + data + cpu+time
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        
        elif self.alg == 'prev_serversLSTM':
            self.state_size  = numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        
        elif self.alg == 'justserviceLSTM':
            self.state_size  = 2*numberofservices+2+1  #current services+ next services + data + cpu+time
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'simpleLSTM':
            self.state_size = 2
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'GAT':
            self.state_size = 16
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverDQN':
            self.state_size  = numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverA2C':
            self.state_size  = numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverLSTM':
            self.state_size  = numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'GCN_DQN':
            self.state_size  =gcn_paramaters['output_dim']   #previews servers + current services+ next services + data + cpu+time
            self.agent =  GCN_DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'GCN':
            self.state_size = 4
            self.agent = GCN_Agent(num_states=self.state_size, num_actions=self.action_size, hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'], epsilon=learning_arguments['epsilon'], maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'direction_nearestserver_prev_servers_plus_service_per_serverDQN':
            self.state_size  =numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
       
        else:
            print("The algorithm is not IMPELEMENTED")
            self.state_size = 2
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        #self.state_size  = 2  #current services+ next services + data + cpu #TODO STATE SIZE
        #self.agent.TrainNet.model.summary()
        #if filename_png:
           # plot_model(self.agent.TrainNet.model, to_file=filename_png+'/learning_model.png', show_shapes=True)

    
    def state(self,task,tasks,done_tasks,deadline,servers_service_info,gat=None,DAG=None,nearest_server_id=None,Embeddings=None,user_direction=None):
        if self.alg == 'simpleDQN' or self.alg == 'simpleA2C' or self.alg == 'simpleLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                state[0] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[1] = task.input_data_length / self.max_data_length
            return state
        
        elif self.alg == 'prev_servers_plus_service_per_serverDQN' or self.alg == 'prev_servers_plus_service_per_serverA2C'  or self.alg == 'prev_servers_plus_service_per_serverLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                server_service = torch.tensor(servers_service_info, dtype=torch.float)
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+server_service.numel()] = server_service.view(-1)
                state[3+server_service.numel():3+server_service.numel()+servers.numel()] = servers
                state[3+server_service.numel()+servers.numel():3+server_service.numel()+servers.numel()+service.numel()] = service
                state[3+server_service.numel()+servers.numel()+service.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()] = succ_services

            return state
        elif self.alg == 'prev_serversDQN' or self.alg == 'prev_serversA2C'  or self.alg == 'prev_serversLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+servers.numel()] = servers
                state[3+servers.numel():3+servers.numel()+service.numel()] = service
                state[3+servers.numel()+service.numel():3+servers.numel()+service.numel()+succ_services.numel()] = succ_services

            return state
        elif self.alg =='justserviceDQN' or self.alg =='justserviceA2C' or self.alg =='justserviceLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0

                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+service.numel()] = service
                state[3+service.numel():3+service.numel()+succ_services.numel()] = succ_services
            return state
        elif self.alg =='GAT':
            state = torch.zeros(self.state_size)
            if task is None:
                return state
            else:
                data = from_networkx(DAG)
                x = [[0, 0, 0, 0]]
                for t in tasks.values():
                    x.append([t.cpu_cycle * 1e-6 / self.max_cpu_cycles, t.input_data_length / self.max_data_length, t.outputlength / self.max_data_length, t.service / self.numberofservices])
                data.x = torch.tensor(x, dtype=torch.float)
                state_comp = gat.encode(data)
                keys_list = list(tasks.keys())
                state = state_comp[keys_list.index(task.task_number)]
                return state
        elif self.alg == 'GCN_DQN':
            if task is None:
                return torch.zeros((self.state_size,))
            else:
                return Embeddings[int(task.task_number)-1]
        elif self.alg == 'GCN':
            state = torch.zeros(self.state_size)
            if task is None:
                return state
            else:
                # Convert DAG to graph representation
                data = from_networkx(DAG)
                
                # Prepare node features
                x = []
                for t in tasks.values():
                    x.append([t.cpu_cycle * 1e-6 / self.max_cpu_cycles, t.input_data_length / self.max_data_length, t.outputlength / self.max_data_length, t.service / self.numberofservices])
                data.x = torch.tensor(x, dtype=torch.float)
                
                # Prepare edge index
                edge_index = data.edge_index
                self.agent.edge_index = edge_index
                
                # Use GCN model to get the state representation
                self.agent.TrainNet.eval()
                with torch.no_grad():
                    output = self.agent.TrainNet.model(data.x, edge_index)
                
                # Find the node index that corresponds to the task_number
                task_number = task.task_number
                node_index = None
                for i, (node, attr) in enumerate(DAG.nodes(data=True)):
                    if node == task_number:
                        node_index = i
                        break
                
                if node_index is None:
                    raise ValueError(f"Task number {task_number} not found in DAG nodes.")
                
                # Use the node index to get the state representation
                state = output[node_index]
                return state
  
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverDQN' or self.alg == 'nearestserver_prev_servers_plus_service_per_serverA2C'  or self.alg == 'nearestserver_prev_servers_plus_service_per_serverLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                server_service = torch.tensor(servers_service_info, dtype=torch.float)
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                nearestserver = torch.zeros(self.numberofservers, dtype=torch.float)
                
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                nearestserver[nearest_server_id] = 1.0
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+server_service.numel()] = server_service.view(-1)
                state[3+server_service.numel():3+server_service.numel()+servers.numel()] = servers
                state[3+server_service.numel()+servers.numel():3+server_service.numel()+servers.numel()+service.numel()] = service
                state[3+server_service.numel()+servers.numel()+service.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()] = succ_services
                state[3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel():] = nearestserver

            return state
        elif self.alg == 'direction_nearestserver_prev_servers_plus_service_per_serverDQN':
            state = torch.zeros(self.state_size)
            if task is not None:
                server_service = torch.tensor(servers_service_info, dtype=torch.float)
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                nearestserver = torch.zeros(self.numberofservers, dtype=torch.float)
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                nearestserver[nearest_server_id] = 1.0
                if user_direction == None:
                    direction = 0.0
                else:
                    if user_direction == "+horizontal":
                        direction = 1.0
                    elif user_direction == "-horizontal":
                        direction = -1.0
                    elif user_direction == "+vertical":
                        direction = 1.0
                    else:
                        direction = -1.0
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+server_service.numel()] = server_service.view(-1)
                state[3+server_service.numel():3+server_service.numel()+servers.numel()] = servers
                state[3+server_service.numel()+servers.numel():3+server_service.numel()+servers.numel()+service.numel()] = service
                state[3+server_service.numel()+servers.numel()+service.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()] = succ_services
                state[3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()+nearestserver.numel()] = nearestserver
                state[3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()+nearestserver.numel()] = direction
            return state
        else:
            #print("The algorithm is not IMPELEMENTED")
            state = torch.zeros(self.state_size)
            if task is not None:
                state[0] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[1] = task.input_data_length / self.max_data_length
            return state
        