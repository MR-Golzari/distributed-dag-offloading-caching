import numpy as np
from results import Results
import networkx as nx

class Task:
    """Task Class

    This class defines a task.

    Attributes
    ----------
        user_id : int
            the id of owner of the task
        task_number : int
            the id of the task
        cpu_cycle : int
            the number of cpu cycles
        data_length : int
            the length of input data
        service : int
            the service id of the task
        predecessors : list[int]
            the list of predecessors of the task
        not_done_predecessors : list[int]
            the list of predecessors that are not executed yet
        successors : list[int]
            the list of successors of the task
        result : Results
            the simulation results of executing the task
        assigned_server : int
            the assigned server to the task
    .. note::
        for now, the service and data_length is assigned randomly
    """
    def __init__(self,user_id,tasknumber,cpu_cycles,service,DAG):
        """Initiates a task object.

        Arg
        ----------
            user_id : int
                The id of the user
            tasknumber : int
                The id of the task
            max_cpu_cycles :
                Maximum cpu cycles of the task
            max_data_length
                Maximum of input data length of the task
            numberofservices
                total number of services
            DAG
                directed cyclic graph of the application

        Returns
        -------
        None

        """

        self.user_id = user_id
        self.task_number = tasknumber
        # assign a random service for each task
        self.cpu_cycle = cpu_cycles*1e6
        self.remained_cpu_cycle = self.cpu_cycle
        # assign data length for each task
        self.outputs_length={}
        self.input_data_length=0
        self.predecessors=[]
        self.not_done_predecessors=[]
        self.successors=[]
        self.done=False
        self.outputlength = 0
        self.DAG = DAG
        if(DAG):
            connected_edges = list(DAG.edges())
            # find the predecessors of the task
            self.predecessors = list(DAG.pred[tasknumber].keys())
            if '0' in self.predecessors:
                self.predecessors.remove('0')
            self.not_done_predecessors = list(DAG.pred[tasknumber].keys())
            if '0' in self.not_done_predecessors:
                self.not_done_predecessors.remove('0')
            # find the successors of the task
            self.successors = list(DAG.succ[tasknumber].keys())
            # Get the values of the edges before the node
            for edge in connected_edges:
                if (edge[1] == tasknumber):
                    if  (edge[0]=='0'):
                        self.input_data_length= DAG[edge[0]][edge[1]]['datalength']
                    else:
                        self.input_data_length = 0
                if (edge[0] == tasknumber):
                    self.outputs_length[edge[1]] = DAG[edge[0]][edge[1]]['datalength']
                    self.outputlength=DAG[edge[0]][edge[1]]['datalength']
        # assign a random service for each task
        self.service = service
        # store the tic tac time for transefering data
        self.reset()
    def reset(self):
        self.result = Results()
        if len(self.predecessors)==0:
            self.result.ready_time = 0
        # assinged server
        self.assigned_server = -1
