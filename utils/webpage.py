import streamlit as st
import pandas as pd
from utils.network import *

def node_num(col):
    node_num = col.number_input('Number of nodes',min_value=1,max_value=100,value=1,key='n')
    return node_num

def edge_input(session,col,node_num:int):
    want_edge = col.multiselect('Select nodes',[i for i in range(node_num)])
    col1,col2,col3 = st.columns([1,1,1])
    if 'edge_list' not in session:
        session.edge_list = []
    session.edge_list = remove_no_exist(node_num,session.edge_list)
    if col1.button('Make Edge'):
        for i in range(len(want_edge)):
            for j in range(i+1,len(want_edge)):
                if (want_edge[i],want_edge[j]) not in session.edge_list:
                    session.edge_list.append((want_edge[i],want_edge[j]))
    
    if col2.button('Remove Edge'):
        for i in range(len(want_edge)):
            for j in range(i+1,len(want_edge)):
                if (want_edge[i],want_edge[j]) in session.edge_list:
                    session.edge_list.remove((want_edge[i],want_edge[j]))
    
    if col3.button('Clear edge'):
        session.edge_list = []

def remove_no_exist(node_num:int,edge_list:list[tuple[int,int]]):
    result = []
    max_node = node_num
    for i in edge_list:
        if max(i) < max_node:
            result.append(i)
    return result

def out_node(node_num:int) -> list[int]:
    if node_num > 0:
        return [0]
    else:
        return []

def pressure(session,out_node):
    st.write('Pressure')
    col1,col2,col3 = st.columns([1,1,1])
    result_dict = {}
    if not out_node or not session.edge_list:
        return None
    index = 0
    for i in session.edge_list:
        if out_node[0] in i:
            if index % 3 == 0:
                value = col1.number_input(f'{i}')
            elif index % 3 == 1:
                value = col2.number_input(f'{i}')
            else:
                value = col3.number_input(f'{i}')
            result_dict[i] = value
            index += 1
        else:
            result_dict[i] = 0
    return result_dict

def plot(session,node_num,edge_list,out_node,P=[],Q:list=[]):
    session.G = network(node_num)
    session.G.create_edge(edge_list)
    if P and Q:
        P = [f'{i:.1f}' for i in P]
        P_list = ['outdoor'] + P
        P_dict = {i:P_list[i] for i in range(len(P_list))}
        o_Q_dict = {edge_list[i]:Q[i] for i in range(len(Q))}
        Q = [f'{i:.1f}' for i in Q]
        Q_dict = {edge_list[i]:Q[i] for i in range(len(Q))}
    else:
        P_dict = {}
        o_Q_dict = {}
        Q_dict = {}
    session.G.show(out_node,P_dict,Q_dict)
    return o_Q_dict

def table_flow(modify_Q_dict):
    df = pd.DataFrame()
    df['Source'] = [i[0] for i in list(modify_Q_dict.keys())]
    df['Target'] = [i[1] for i in list(modify_Q_dict.keys())]
    flow_list = []
    for i in modify_Q_dict.values():
        # i = round(float(i),2)
        flow_list.append(f'{i:.3f}')
    df['Flow (kg/s)'] = flow_list
    df.sort_values(by=['Source','Target'],inplace=True)
    return df.T