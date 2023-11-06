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

def pressure(session,out_node:list[int]):
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

def node_setting(session,node_num:int,out_node:list[int]):
    result_dic = {}
    people_num = {}
    concentration  = {}
    volume = {}
    st.write('Node setting')
    col1,col2,col3 = st.columns([1,1,1])
    if not out_node or not session.edge_list:
        return None
    index = 0
    for i in range(node_num):
        if i == out_node[0]:
            p_value = 0
            c_value = col1.number_input(label='Outdoor ppm',value=400,key = i )
            v_value = 1e20
        elif index % 3 == 0:
            ccol1,ccol2,ccol3 = col1.columns([1,1,1])
            p_value = ccol1.number_input(f'N{i} num', step=1 ,format='%i')
            c_value = ccol2.number_input(label= 'ppm',value=440, key = i)
            v_value = ccol3.number_input(label='m3', key = f'v{i}')
        elif index % 3 == 1:
            ccol1,ccol2,ccol3 = col2.columns([1,1,1])
            p_value = ccol1.number_input(f'N{i} num', step=1 ,format='%i')
            c_value = ccol2.number_input(label= 'ppm',value=440, key = i)
            v_value = ccol3.number_input(label='m3', key = f'v{i}')
        else:
            ccol1,ccol2,ccol3 = col3.columns([1,1,1])
            p_value = ccol1.number_input(f'N{i} num', step=1 ,format='%i')
            c_value = ccol2.number_input(label= 'ppm',value=440, key = i)
            v_value = ccol3.number_input(label='m3', key = f'v{i}')

        people_num[i] = p_value
        concentration[i] = c_value
        volume[i] = v_value
        index += 1
    result_dic['people_num'] = people_num
    result_dic['concentration'] = concentration
    result_dic['volume'] = volume
    return result_dic

def time_setting(session) ->tuple[float,float]:
    if not out_node or not session.edge_list:
        return None,None
    col1,col2 = st.columns([1,1])
    total_time = col1.number_input(label='Total Time (s)',step=0.01, format='%f')
    delta_time = col2.number_input(label='Delta Time (s)',step=0.01, format='%f')
    return total_time,delta_time

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
        i = round(float(i),3)
        flow_list.append(i)
    df['Flow (m3/s)'] = flow_list
    df.sort_values(by=['Source','Target'],inplace=True)
    return df.T

def table_concentration(c_series,node_num:int):
    if c_series.any():
        df = pd.DataFrame(c_series)
        columns = ['Time (s)'] + [f'Node{i}' for i in range(node_num)]
        df.columns = columns
        st.dataframe(df,use_container_width=True)