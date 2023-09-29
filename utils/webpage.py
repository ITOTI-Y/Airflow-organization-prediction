# # # 函数
# # def create_node(G,num):
# #     if G.number_of_nodes() > num:
# #         G.clear()
# #     for i in range(num):
# #         G.add_node(i)

# # def edge_group(G,want_edge):
# #     result = []
# #     for i in range(len(want_edge)):
# #         for j in range(i+1,len(want_edge)):
# #             result.append((want_edge[i],want_edge[j]))
# #     for i in result:
# #         G.add_edges_from([i])

# # def remove_edge(G,want_edge):
# #     result = []
# #     for i in range(len(want_edge)):
# #         for j in range(i+1,len(want_edge)):
# #             result.append((want_edge[i],want_edge[j]))
# #     for i in result:
# #         G.remove_edges_from([i])

# # def choose_vent(G,node_list:list):
# #     node_list = node_list
# # # 创建两个平行的列表
# #     col1,col2 = st.columns(2)
# #     col1.write('Input Nodes')
# #     col2.write('Output Nodes')
# #     with col1:
# #         input_nodes = []
# #         for node in node_list:
# #             if st.checkbox(f'{node}',key=f'input{node}'):
# #                 input_nodes.append(node)
# #     with col2:
# #         remain_node = [i for i in node_list if i not in input_nodes]
# #         output_nodes = []
# #         for node in remain_node:
# #             if st.checkbox(f'{node}',key=f'output{node}'):
# #                 output_nodes.append(node)
# #     edge_group(G,output_nodes+input_nodes)
# #     return input_nodes,output_nodes

# # def color_map(node_list,input_node,output_node):
# #     result = []
# #     for node in node_list:
# #         if node in input_node:
# #             result.append("red")
# #         elif node in output_node:
# #             result.append("blue")
# #         else:
# #             result.append("white")
# #     return result

# # def edge(node_num):
# #     want_edge = st.multiselect('Select nodes',[i for i in range(node_num)])

# #     col1,col2,col3 = st.columns([1,1,1])
# #     with col1:
# #         if st.button('Make Edge'):
# #             edge_group(st.session_state.G,want_edge)
# #     with col2:
# #         if st.button('Remove Edge'):
# #             remove_edge(st.session_state.G,want_edge)
# #     with col3:
# #         if st.button('Clear edge'):
# #             st.session_state.G.clear_edges()

# # def edge_labels(edges):
# #     result = {edges[i]:i for i in range(len(edges))}
# #     return result

# # 初始化
# if 'G' not in st.session_state:
#     st.session_state.G = nx.DiGraph()

# # 输入参数

# node_num = st.number_input('Number of nodes',min_value=1,max_value=100,value=1,key='n')


# # 网络图界面
# options = {
#     "font_size": 12,
#     "node_size": 300,
#     "edgecolors": "black",
#     "linewidths": 2,
#     "width": 2,
#     "with_labels": True,
#     "connectionstyle": "arc3,rad=0.1",
#     # "with_labels": True,
# }
# fig,ax = plt.subplots()
# create_node(st.session_state.G,node_num)
# edge(node_num)
# pos = nx.shell_layout(st.session_state.G)
# node_list = st.session_state.G.nodes

# input_node,output_node = choose_vent(st.session_state.G,node_list)
# color = color_map(node_list,input_node,output_node)
# # edge_labels_list = edge_labels(st.session_state.G.edges)

# # st.write(edge_labels_list)

# nx.draw(st.session_state.G,pos,node_color=color,**options)
# # nx.draw_networkx_edge_labels(st.session_state.G, pos, edge_labels=edge_labels)

import streamlit as st
import numpy as np

class webpage:
    node_num = 1
    edge_list = []

    def __init__(self):
        pass

    def node_num_input(self):
        self.node_num = st.number_input('Number of nodes',min_value=1,max_value=100,value=1,key='n')
        print(self.node_num)
        return self.node_num
    
    def edge_input(self):
        want_edge = st.multiselect('Select nodes',[i for i in range(self.node_num)])
        col1,col2,col3 = st.columns([1,1,1])
        if col1.button('Make Edge'):
            for i in range(len(want_edge)):
                for j in range(i+1,len(want_edge)):
                    if (want_edge[i],want_edge[j]) not in self.edge_list:
                        self.edge_list.append((want_edge[i],want_edge[j]))
        
        if col2.button('Remove Edge'):
            for i in range(len(want_edge)):
                for j in range(i+1,len(want_edge)):
                    if (want_edge[i],want_edge[j]) in self.edge_list:
                        self.edge_list.remove((want_edge[i],want_edge[j]))
        
        if col3.button('Clear edge'):
            self.edge_list = []

        self.remove_no_exist()
        print(self.edge_list)

    def remove_no_exist(self):
        result = []
        max_node = self.node_num
        for i in self.edge_list:
            if max_node not in i:
                result.append(i)
        self.edge_list = result

    def output_node(self) -> list[int]:
        col1,col2,col3 = st.columns([1,1,1])
        result = []
        for i in range(self.node_num):
            if i % 3 == 0:
                if col1.checkbox(f'{i}'):
                    result.append(i)
            elif i % 3 == 1:
                if col2.checkbox(f'{i}'):
                    result.append(i)
            else:
                if col3.checkbox(f'{i}'):
                    result.append(i)
        return result
