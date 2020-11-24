def QLearn(model, maze):
    print("TODO")


class QLearnMem:
    def __init__(self, memSize = 10):
        self.size = memSize
        self.memory = []
    
    
    def append(self, item):
        self.memory.append(item)
        if len(self.memory) > self.size:
            self.memory.remove(0)   