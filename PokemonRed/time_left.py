class TimeLeft:
    def __init__(self, time_left=0):
        self.time_left = 0
        self.time_passed = 0
        
    def get_time_left(self):
        return self.time_left
    
    def set_time_left(self, time_left):
        self.time_left = time_left
    
    def decrease_time_left(self, time_passed):
        self.time_left -= time_passed
        self.time_passed += time_passed
