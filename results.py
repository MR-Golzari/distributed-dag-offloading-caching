class TimeTicTac: # a class to set the start and end time of the task and calculate duration
    def __init__(self):
        self.start=-1
        self.end = -1
        self.duration = 0
    def set_start(self,t):
        self.start=t

    def set_end(self,t):
        self.end=t
        self.duration = self.end - self.start
class Results:
    def __init__(self):
        self.transfer_time = TimeTicTac()
        # store the tic tac time for waiting in the queue
        self.waiting_time = TimeTicTac()
        # store the tic tac time for processing
        self.processing_time = TimeTicTac()

        self.finish_time = float('inf')
        self.pred_latency = float('inf')
        self.service_latency = float('inf')
        self.data_transfer_latency = float('inf')
        self.computing_latency = float('inf')
        self.ready_time = float('inf')