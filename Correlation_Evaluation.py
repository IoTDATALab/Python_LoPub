import numpy
import graph
import random
import csv
import Get_Params
import Get_Rappor
import Dependency
import os
#import copy
import time
import pickle
from copy import copy
from Estimate_Joint_Distribution import att_combin, estimate_2d,\
    list_product, rappor_list_paste, true_joint_distribution
from JunctionTree import cliques_to_locs,independe_draw,conditional_draw,independe_draw2,conditional_draw2
from matplotlib.font_manager import pickle_dump
from collections import Counter
import Evaluation_SVM
import gc
from numpy import reshape


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################        
Changeflag=True
ISflag=1

curr_time=


file_id=2
fai_C=0.55   #from 0.2, 0.3, 0.4, 0.5
#f=0.5 # from 0.1, 0.2, 0.3, 0.4, 0.5  *********
# bloombit=128
# hashbit=16
bloombit=32
hashbit=4
dt=0.01
readlimit=80000
samplerate=0.1  # from 0.01, 0.05, 0.1, 0.5, 1
sparse_rate=0.0
if sparse_rate==0.0:
    get_rid_flag=False
else:
    get_rid_flag=True
for file_id in [2,3,4]:
    if file_id==4:
        fai_list=[0.2]
        col_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    if file_id==2:
        bloombit=128
        hashbit=4
        fai_list=[0.3]
        col_list=[9,14]
        #col_list=[0,1,2,3,4,5,6,7,8,10,11,12,13]
    if file_id==3:
        bloombit=32
        hashbit=4
        fai_list=[0.2]
        col_list=[2,9,22,23]
    for fai_C in fai_list:
        
        for f in [0.1, 0.3, 0.5, 0.7, 0.9]:
   
            param_string='D_'+str(file_id)+'_C_'+str(fai_C)+'_f_'+str(f)+'_B_'+str(bloombit)+'_H_'+str(hashbit)+'_S_'+str(samplerate)+'_R_'+str(readlimit)
            #print(param_string)
            os.chdir('C:\Users\Ren\workspace2\DisHD\output')
           
            folder_h = param_string
            isExists=os.path.exists(folder_h)
            if not isExists:
                os.makedirs(r'%s/%s'%(os.getcwd(),folder_h))
                os.chdir(folder_h)
            else:
                os.chdir(folder_h)
            isHave1=os.path.exists('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs.pickle')
            isHave2=os.path.exists('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs2.pickle')
            #print (isHave)
            if Changeflag or (not isHave1) :
            
                folder_l = time.strftime(r"%Y-%m-%d_%H-%M-%S",time.localtime())
                os.makedirs(r'%s/%s'%(os.getcwd(),folder_l))
                folder=folder_l
                
                att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(file_id, readlimit, samplerate)
                bit_cand_list, bit_list, bitsum_list=Get_Rappor.rappor_process(bloombit, hashbit, f, att_num, node_num, true_node_num, rowlist, multilist, file_id)
                #att_num,node_num,true_node_num,rowlist,multilist,bit_cand_list,bit_list,bitsum_list=Get_Rappor.Get_rid_sparse(file_id, readlimit, samplerate, bloombit, hashbit, f, sparse_rate,get_rid_flag)
                
                
                
                freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num, node_num, rowlist, multilist)

                TrueDepG,Truens,True_CorMat,freqrate2,freqrate1=Dependency.True_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,fai_C)
                TrueDG=numpy.array(TrueDepG)

                TrueNS=numpy.array(Truens)
                [True_TrG,True_jtree, True_root, True_cliques, True_B, True_w]=graph.graph_to_jtree(TrueDG,TrueNS)
                True_TrG=numpy.array(True_TrG)
                
                #print(True_jtree)
                #print('true prob:',freqrate1)  
                #exit(0)
       
                Corr_Matrix,DepenGraph,ns,att_num,node_num,origin_node_num,row_list,multilist,bit_cand_list, bit_list,bitsum_list,p_comb_list,p_single_list=Dependency.Get_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,bloombit, hashbit, f,bit_cand_list,bit_list,bitsum_list,fai_C)
                DG=numpy.array(DepenGraph)
                [TrG,jtree, root, cliques, B, w]=graph.graph_to_jtree(DG,TrueNS)
                #print(DG)
                DGrate=Counter(reshape(DG-TrueDG,att_num*att_num))
                print('True Cliques',True_cliques)
                print('Cliques:',cliques)
                print(DGrate)
                DGrr=DGrate[0]/(1.0*TrueDG.size)
                DGfp=DGrate[1]/(1.0*TrueDG.size)
                DGtn=DGrate[-1]/(1.0*TrueDG.size)
                
                write_list=[[fai_C,f,DGrr,DGfp,DGtn,sparse_rate,samplerate]]
                print(write_list)
                os.chdir('C:\Users\Ren\workspace2\DisHD\output')
                with open('file-'+str(file_id)+'-Correlation_EM.csv','a') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(write_list)