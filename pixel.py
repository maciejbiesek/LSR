import numpy as np

class Neigh:
    
    def __init__(self, color):
        self.color = color
        self.distance = 0.0
        
    def __str__(self):
        return str(self.color) + " " + str(self.distance)
    
    def get_dist(self, color):
        self.distance = np.linalg.norm(self.color - color)
        

class Pixel:
    
    def __init__(self, color):
        self.color = color
        self.neighbourhood = []
        
    def add_neighbour(self, pixel):
        self.neighbourhood.append(pixel)
        
    def sort(self):
        self.neighbourhood.sort(key=lambda x: x.distance, reverse=True)
            