import streamlit as st
from utils.web.network import *
from utils.web.webpage import *
from utils.web.Forward_Calculation import *
from matplotlib import pyplot as plt
from utils.module.UNet_v2 import *
from utils.module.datasets import Image_dataset
from utils.module.train import Train
from utils.module.postprocessing import Postprocessing
import numpy as np
from torchvision.transforms import v2

image = image_sidebar(r'data\train_image')

# 模型预测
if image != None:
    image = v2.ToTensor()(image)
    image = v2.Resize((512,512),antialias=True)(image)
    n_classes = 4
    model = UNetV2(n_classes=n_classes, deep_supervision=True ,pretrained_path=None)
    masks = model.predict(image,model_path=r'result\best-0.97.pth')
    # 图像后处理
    post = Postprocessing(image,masks)
    connections = post.find_connected_islands(show=True)
    st.image(r'./result/picture/label_island.jpg')
    st.session_state['image'] = image
    st.session_state['connections'] = connections
    st.session_state['masks'] = masks

    position_num = len(connections.keys())

else:
    position_num = 1
    connections = None

col1,col2 = st.columns([1,1])
node_num = node_num(col1,position_num)
out_node = out_node(node_num)
edge_input(st.session_state,col2,node_num,predict_connection=connections)
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
    if total_t != 0 and delta_t != 0:
        c_series = cal.main()
modify_Q_dict = plot(st.session_state,node_num,edge_list,out_node,P,Q)



# 网页界面
st.title('Ventilation Network')
col3,col4 = st.columns([1,1])
st.pyplot(fig)
st.dataframe(table_flow(modify_Q_dict),use_container_width=True)
table_concentration(c_series,node_num)