import numpy as np
from scipy.linalg import solve

class VNetwork:

    def __init__(self,node_num:int,edges:list,out_node:list[int],pressure_dict:dict):
        self.node_num = node_num
        self.edges = edges
        self.out_node = out_node
        self.pressure_dict = pressure_dict
        self.R = np.ones((len(edges), 1)) * 0.5
        self.P = np.random.rand(node_num, 1) * 10
        self.P = self.P[1:]
        self.rho = np.ones((len(edges), 1))

    def flow_equation(self,BP,H,R):
        return np.sign(BP + H) * np.sqrt(np.abs(BP + H) / R)
    
    def make_B(self):
        edges = self.edges
        node_num = self.node_num
        B = np.zeros((node_num,len(edges)))
        for i in range(len(edges)):
            B[edges[i][0],i] = 1
            B[edges[i][1],i] = -1
        return B[1:].T
    
    def jacobian(self,B,BP,H,R):
        diag_elements = 0.5 / (np.sqrt(R * np.abs(BP + H))).flatten()
        J = np.dot(B.T, B * diag_elements[:, np.newaxis])
        return J
    
    def newton_method(self,tol=1e-6, max_iter=1000):
        P = self.P
        B = self.make_B()
        H = np.array([self.pressure_dict[edge] for edge in self.edges]).reshape(-1, 1)
        R = self.R
        for _ in range(max_iter):
            BP = np.dot(B, P)
            Q = self.flow_equation(BP, H, R)
            J = self.jacobian(B, BP, H, R)
            delta_P = solve(J, np.dot(B.T, Q))
            P -= delta_P
            
            if np.linalg.norm(delta_P) < tol:
                break
        P = P.tolist()
        P = [i[0] for i in P]
        Q = Q.tolist()
        Q = [i[0] for i in Q]
        
        return P,Q

