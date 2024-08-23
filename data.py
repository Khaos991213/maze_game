class dat():
    def __init__(self):
        self.state_array = []
        self.action_array = []
       
    def AddAction(self, action):
        self.action_array.append(action)
    def AddState(self, state):
        self.state_array.append(state)
    def get_state(self):
        return self.state_array