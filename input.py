INPUT_DICT={
    'Folder':'prev_servers_plus_service_per_serverDQN',
    'alg':'nearestserver_prev_servers_plus_service_per_serverDQN', # {direction_nearestserver_prev_servers_plus_service_per_serverDQN,GCN_DQN,GCN,nearestserver_prev_servers_plus_service_per_serverDQN,prev_servers_plus_service_per_serverDQN,GAT,prev_servers_plus_service_per_serverDQN, simpleDQN,prev_serversDQN,justserviceDQN,simpleA2C,prev_serversA2C,justserviceA2C, prev_serversLSTM,simpleLSTM,justserviceLSTM}
    'comment':'',
    'SimulationTime':1,
    'Number of users': 20,
    'Number of servers':10,
    'Number of tasks for each user': 10,
    'Number of services':10,
    'server capacity': 2, #Ks 
    'seed': 30,
    'xlim':1000,
    'ylim':1000,
    'max_cpu_cycles':200,  #Mcycle
    'max_data_length':1500e3, #bits 100e3
    'min_cpu_frquency':0.2,  #Gcycle/sec
    'max_cpu_frquency':30, #Gcycle/sec
    'min_service_data_length':10,  #Mbits
    'max_service_data_length':1000,  #Mbits

    'min_rate_between_servers': 10,  # Mbit/sec  fixed
    'max_rate_between_servers': 40,  # Mbit/sec fixed

    'min_load_on_server': 0.2, #Mcycle
    'max_load_on_server': 100, #Mcycle

    'Min rate to cloud': 1, #Mbit/sec  fixed
    'Max rate to cloud': 3, #Mbit/sec fixed
    'Power': 10000, #wat
    'Bandwidth': 1000000, #Hz
    'Number of episodes': 150000,
    'Number of runs':1,
    'filling steps':500,
    'steps to updates':100,
    'deadline':1,
    'update deadline': True,
    'beta': 0.1, #transfer learning parameter

    'velocity': 0, #km/h
    'federated_learning_param_server': 0.0,
    'caching decision enabled': True
}
learning_arg={
    'batch_size':1024,
    'learning_rate':0.001,
    'hidden_units': [64],
    'gamma': 0.9,
    'max_experiences':10000,
    'min_experiences':1024,
    'epsilon':0.001,
    #'epsilon':0.0,
    'maximum_exploration': 20000,  # Maximum exploration step
}
GCN_paramaters={
    'layers': [64,64],
    'dropout': 0.5,
    'output_dim': 1,
    'learning_rate':0.001,
}