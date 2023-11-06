import numpy as np
from scipy.linalg import solve
import collections

class VNetwork:

    def __init__(self,node_num:int,edges:list,out_node:list[int],pressure_dict:dict):
        self.node_num = node_num
        self.edges = edges
        self.out_node = out_node
        self.pressure_dict = pressure_dict
        self.R = np.ones((len(edges), 1)) * 0.5
        self.P = np.random.rand(node_num, 1) * 10
        self.P = self.P
        self.rho = np.ones((len(edges), 1))

    def flow_equation(self,BP,H,R):
        return np.sign(BP + H) * np.sqrt(np.abs(BP + H) / R)
    
    def make_B(self):
        """
        B: 节点-1（行） x 边（列） （节点联系矩阵）
        """
        edges = self.edges
        node_num = self.node_num
        B = np.zeros((node_num,len(edges)))
        for i in range(len(edges)):
            B[edges[i][0],i] = 1
            B[edges[i][1],i] = -1
        return B
    
    def jacobian(self,B,BP,H,R):
        diag_elements = 0.5 / (np.sqrt(R * np.abs(BP + H))).flatten()
        J = np.dot(B.T, B * diag_elements[:, np.newaxis])
        return J
    
    def newton_method(self,tol=1e-6, max_iter=2000):
        P = self.P[1:]
        B = self.make_B()[1:].T
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
    
class Cal_concentration:
    CP = 40000 # 人体呼出二氧化碳浓度 ppm
    VP = 1.5e-4 # 人体呼出气体排放速率 m3/s

    def __init__(self,B,Q,node_info,total_t:float | int,delta_t:float | int):
        self.B = B
        self.Q = Q
        self.N = np.array(list(node_info['people_num'].values()))
        self.V = np.array(list(node_info['volume'].values()))
        self.C0 = np.array(list(node_info['concentration'].values()))
        self.total_t = total_t
        self.delta_t = delta_t
        self.S = self.source()
        C = self.set_c_matrix(self.C0)

    def source(self):
        result = self.N * self.CP * self.VP
        return result
    
    def set_c_matrix(self,C0):
        BQ = self.B * self.Q
        result = np.zeros_like(BQ)
        for row in range(BQ.shape[0]):
            for i in range(BQ.shape[1]):
                value = BQ[row,i]
                if BQ[row,i] != 0:
                    c = C0[np.argmax(BQ[:,i])]
                    result[row,i] = value * c
        result =  - np.sum(result,axis=1)
        result[0] = 0
        return result
    
    def main(self):
        step = int(self.total_t // self.delta_t)
        result = np.zeros((step,self.B.shape[0]))
        c_o = self.C0
        time_series = np.arange(0,self.total_t,self.delta_t)
        for i in range(step):
            qc = self.set_c_matrix(c_o)
            dc = (qc + self.S) / self.V
            c_n = c_o + dc * self.delta_t
            result[i] = c_n
            c_o = c_n
        result = np.concatenate((time_series.reshape(-1,1),result),axis=1)
        return result



if __name__ == "__main__":
    node_num = 4
    edge_list = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]
    out_node = [0]
    pressure_dict = {(0, 1): 1.0, (0, 3): 2.0, (1, 2): 0, (1, 3): 0, (2, 3): 0}
    a = VNetwork(node_num,edge_list,out_node,pressure_dict)
    P,Q = a.newton_method()
    B = a.make_B()

    people_num = {0:0,1:1,2:2,3:3}
    volume = {0:1e20,1:45,2:45,3:45}
    concentration = {0:400,1:440,2:440,3:440}
    node_info = {'people_num':people_num,
                 'volume':volume,
                 'concentration':concentration}
    total_t = 1000
    delta_t = 10
    cal = Cal_concentration(B,Q,node_info,total_t,delta_t)
    cal.main()