import pickle
import os

ISflag=1
file_id=10
fai_C=0.45
f=0.6
bloombit=16
hashbit=2
dt=0.01
readlimit=50000
samplerate=0.01



param_string='D_'+str(file_id)+'_C_'+str(fai_C)+'_f_'+str(f)+'_B_'+str(bloombit)+'_H_'+str(hashbit)+'_S_'+str(samplerate)+'_R_'+str(readlimit)
print(param_string)
os.chdir('C:\Users\Ren\workspace2\DisHD\output')
folder_h = param_string
with open('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs.pickle') as f:
    att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,bit_cand_list,bit_list,bitsum_list,TrueDepG,Truens,True_CorMat,True_jtree, True_root, True_cliques, True_B, True_w,Corr_Matrix,DepenGraph,ns,p_comb_list,p_single_list,jtree, root, cliques, B, w,new_data_list= pickle.load(f)
    
    
print(att_num,node_num,true_node_num,rowlist,freqrate2)