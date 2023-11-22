import streamlit as st
from utils.network import *
from utils.webpage import *
from utils.Forward_Calculation import *
from matplotlib import pyplot as plt

col1,col2 = st.columns([1,1])
node_num = node_num(col1)
out_node = out_node(node_num)
edge_input(st.session_state,col2,node_num)
edge_list = st.session_state.edge_list
pressure_dict = pressure(st.session_state,out_node)


fig,ax = plt.subplots()
P,Q = [],[]

node_info = node_setting(st.session_state,node_num,out_node)
total_t,delta_t = time_setting(st.session_state)
c_series = np.empty(0)

if st.button('Calculate'):
    network = VNetwork(node_num,edge_list,out_node,pressure_dict)
    P,Q = network.newton_method(max_iter=2000)
    B = network.make_B()
    cal = Cal_concentration(B,Q,node_info,total_t,delta_t)
    c_series = cal.main()
modify_Q_dict = plot(st.session_state,node_num,edge_list,out_node,P,Q)



# 网页界面
st.title('Ventilation Network')
col3,col4 = st.columns([1,1])
st.pyplot(fig)
st.dataframe(table_flow(modify_Q_dict),use_container_width=True)
table_concentration(c_series,node_num)