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


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################        
Changeflag=True
ISflag=1


file_id=4
fai_C=0.55   #from 0.2, 0.3, 0.4, 0.5
#f=0.5 # from 0.1, 0.2, 0.3, 0.4, 0.5  *********
# bloombit=128
# hashbit=16
bloombit=128
hashbit=16
dt=0.01
readlimit=50000
samplerate=0.1   # from 0.01, 0.05, 0.1, 0.5, 1
sparse_rate=0.0
if sparse_rate==0.0:
    get_rid_flag=False
else:
    get_rid_flag=True
for file_id in [2,4,2]:
    if file_id==4:
        fai_list=[0.2,0.99]
        col_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    if file_id==2:
        fai_list=[0.5,0.99]
        col_list=[9,14]
    if file_id==3:
        fai_list=[0.5,0.99]
        col_list=[2,9,22,23]
    for fai_C in fai_list:
        
        for f in [0.1,0.3,0.5,0.7,0.9,0.92,0.94,0.96,0.98]:
                
            
            #############################################################################################################################################################################################################
            #############################################################################################################################################################################################################
            
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
                
                att_num,node_num,true_node_num,rowlist,multilist,bit_cand_list,bit_list,bitsum_list=Get_Rappor.Get_rid_sparse(file_id, readlimit, samplerate, bloombit, hashbit, f, sparse_rate,get_rid_flag)
                
                
                #att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(file_id,readlimit,samplerate)
                
                freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num, node_num, rowlist, multilist)
                #bit_cand_list,bit_list,bitsum_list=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num,node_num,true_node_num,rowlist,multilist)           #(file_id,readlimit,samplerate, bloombit, hashbit, f)
                
                #TrueDepG,Truens,True_CorMat,freqrate2,freqrate1=Dependency.True_Dep_Graph(file_id, readlimit,samplerate, fai_C)
                TrueDepG,Truens,True_CorMat,freqrate2,freqrate1=Dependency.True_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,fai_C)
                TrueDG=numpy.array(TrueDepG)
                #print(True_CorMat)
                #print(TrueDG)
                TrueNS=numpy.array(Truens)
                [True_jtree, True_root, True_cliques, True_B, True_w]=graph.graph_to_jtree(TrueDG,TrueNS)
                print(True_cliques)
                #print(True_jtree)
                print('true prob:',freqrate1)  
                #exit(0)
                ########################################################################## Write Into Files#############################################################################
                
                with open(folder+'\\'+'1-CorrMat_R.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(True_CorMat)
                    
                with open(folder+'\\'+'2-DGraph_R.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(TrueDepG)
                    
                with open(folder+'\\'+'3-JTree_R.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(True_jtree)
                    
                with open(folder+'\\'+'4-Cliques_R.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(True_cliques)
                
                with open(folder+'\\'+'5-JDistribution2_R.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(freqrate2)
            
                with open(folder+'\\'+'5-JDistribution1_R.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(freqrate1)
                
                ##################################################################################################################################################################################################
                #Corr_Matrix,DepenGraph,ns,att_num,node_num,origin_node_num,row_list,multilist,bit_cand_list, bit_list,bitsum_list,p_comb_list,p_single_list=Dependency.Get_Dep_Graph(file_id,readlimit,samplerate,bloombit,hashbit,fai_C, f)
                Corr_Matrix,DepenGraph,ns,att_num,node_num,origin_node_num,row_list,multilist,bit_cand_list, bit_list,bitsum_list,p_comb_list,p_single_list=Dependency.Get_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,bloombit, hashbit, f,bit_cand_list,bit_list,bitsum_list,fai_C)
                DG=numpy.array(DepenGraph)
                #print(DG)
                #print(DG-TrueDG)
                ns=numpy.array(ns)
                [jtree, root, cliques, B, w]=graph.graph_to_jtree(DG,ns)
                print(cliques)
                
                ############################################################ Write into files ############################################################################
                with open(folder+'\\'+'0-Infomation.txt','wb') as fid:
                    info_list=[['att_num',att_num],['node_num',node_num],['origin_node_num',origin_node_num],['sample_rate',samplerate],['file_id',file_id],['fai_C',fai_C],['f',f],['bloombit',bloombit],['hashbit',hashbit]]
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(info_list)
                    
                with open(folder+'\\'+'0-Domain.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(row_list)
                    
                with open(folder+'\\'+'0-DataSet.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(multilist)
                
                with open(folder+'\\'+'1-CorrMat_E.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(Corr_Matrix)
                   
                with open(folder+'\\'+'2-DGraph_E.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(DepenGraph)
                    
                with open(folder+'\\'+'3-JTree_E.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(jtree)
                
                with open(folder+'\\'+'4-Cliques_E.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(cliques)
                
                with open(folder+'\\'+'5-JDistribution1_E.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(p_single_list)
                
                #print(p_comb_list)
                p_comb_len=len(p_comb_list)
                pp=[[]for i in range(p_comb_len)]
                for i in range(p_comb_len):
                    for p_sub in p_comb_list[i]:
                        pp[i].extend(p_sub)
                
                with open(folder+'\\'+'5-JDistribution2_E.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(pp)
                ##########################################################################################################################################################
               
                with open('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs.pickle', 'w') as memoryfile:
                    pickle.dump([att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,bit_cand_list,bit_list,bitsum_list,TrueDepG,Truens,True_CorMat,True_jtree, True_root, True_cliques, True_B, True_w,Corr_Matrix,DepenGraph,ns,p_comb_list,p_single_list,jtree, root, cliques, B, w ], memoryfile)
            
                del freqrow1, freqrow2, TrueDepG, Truens,True_CorMat,True_jtree,True_root,True_cliques,True_B,True_w,newlist
            ####################################################################################################################################################################################################################
            ####################################################################################################################################################################################################################
                #exit(0)    
                new_data_list=[['0' for i in range(att_num)]for j in range(origin_node_num)]
                ############################    EXAMPLE ####################################################################
                cliques_list=cliques_to_locs(cliques)
                
                print(cliques_list)
                sampled_set=set()
                unsampl_list=copy(cliques_list)
                
                while len(unsampl_list)>0 :
                    get_one=unsampl_list.pop()
                    print('computer new', get_one)
                    #independe_draw(ISflag,new_data_list, origin_node_num, get_one, bit_list, bit_cand_list, row_list,p_single_list,p_comb_list, f, dt)
                    independe_draw2(ISflag,new_data_list, origin_node_num, get_one, bit_list, bitsum_list,bit_cand_list, row_list,p_single_list,p_comb_list, f, dt)
                    #print('true:',true_joint_distribution(multilist,row_list,get_one))
                    sampled_set=sampled_set|set(get_one)
                    visited_list=[]
                    while 1 :
                        for clique in unsampl_list:
                            condition_set=set(clique)&set(sampled_set)
                            visited_list.append(clique)
                            if len(condition_set)>0 :
                                if  set(clique).issubset(sampled_set):
                                    print (clique,'already there!')
                                    unsampl_list.remove(clique)
                                    visited_list=[]
                                else:
                                    unsampl_list.remove(clique)
                                    visited_list=[]
                                    condition_list=list(condition_set)
                                    condition_list.sort()
                                    print('computer condition', condition_list, clique)
                                    #conditional_draw(ISflag,new_data_list, origin_node_num, condition_list, clique, bit_list, bit_cand_list, row_list,p_single_list,p_comb_list, f, dt)
                                    conditional_draw2(ISflag,new_data_list, origin_node_num, condition_list, clique, bit_list,bitsum_list, bit_cand_list, row_list,p_single_list,p_comb_list, f, dt)
                                    #print('true:',true_joint_distribution(multilist,row_list,clique))
                                    sampled_set=sampled_set|set(clique) 
                            visit=visited_list
                            unsample=unsampl_list
                            if visit==unsample:
                                break
                        visit=visited_list
                        unsample=unsampl_list
                        if visit==unsample:
                            break
                ############################################################ Write into files ############################################################################
                with open(folder+'\\'+'0-DataSyn.csv','wb') as fid:
                    fid_csv = csv.writer(fid)
                    fid_csv.writerows(new_data_list)
                ##########################################################################################################################################################
                #time_string=time.strftime(r"%Y-%m-%d_%H-%M-%S",time.localtime())
                with open('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs2.pickle', 'w') as memoryfile2:
                    pickle.dump([new_data_list ], memoryfile2)
            
            else:
                print('This has been calculated! Please change the parameters! ')
                
                if (isHave2):
                    with open('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs.pickle') as memoryfile1:
                        att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,bit_cand_list,bit_list,bitsum_list,TrueDepG,Truens,True_CorMat,True_jtree, True_root, True_cliques, True_B, True_w,Corr_Matrix,DepenGraph,ns,p_comb_list,p_single_list,jtree, root, cliques, B, w,new_data_list= pickle.load(memoryfile1)
                else:   
                    with open('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs.pickle') as memoryfile1:
                        att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,bit_cand_list,bit_list,bitsum_list,TrueDepG,Truens,True_CorMat,True_jtree, True_root, True_cliques, True_B, True_w,Corr_Matrix,DepenGraph,ns,p_comb_list,p_single_list,jtree, root, cliques, B, w= pickle.load(memoryfile1)
                
                    with open('C:\Users\Ren\workspace2\DisHD\output\\'+folder_h+'\\objs2.pickle') as memoryfile2:
                        new_data_list= pickle.load(memoryfile2)
                
            
            
            #att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(2,50000,0.1)
            
            
            
            #multilist2,node_num2,att_num2=Get_newdata(20,true_node_num, 0.1)
            n=int(samplerate*node_num)
            sample_list=[]
            sample_order=random.sample(range(node_num),n)
            for i in sample_order:
                    sample_list.append(new_data_list[i])    
            
            ratio=0.8
            loop_time=20
            #col=1
            m1=0.0
            m2=0.0
            leng=0
            #col_list=range(att_num)
            for col in col_list:
                train_x,train_y,test_x,test_y,single_err=Evaluation_SVM.Data_construct(multilist, col, ratio)
            #     if (not single_err):
            #         test_y[random.randint(0,len(test_y))]=1-test_y[0]
                train_x2,train_y2,test_x2,test_y2,single_err2=Evaluation_SVM.Data_construct(sample_list, col, ratio)
            #     if (not single_err2):
            #         test_y2[random.randint(0,len(test_y))]=1-test_y2[0]
                if ((not single_err)):
                    print(col,'omit!'),
                    continue
                else:
                    leng=leng+1
                    t1=Evaluation_SVM.SVM_ratio(train_x, train_y, test_x, test_y, loop_time)
                    m1=m1+t1
                    if (not single_err2):
                        t2=0.5
                    else:
                        t2=Evaluation_SVM.SVM_ratio(train_x2, train_y2, test_x, test_y, loop_time)
                    m2=m2+t2
                    print (' ')
                    print('col:',col,' ',t1,t2)
            print ('file:',file_id,'f:',f,m1/leng,m2/leng)
            write_list=[[fai_C,f,m1/leng,m2/leng,sparse_rate]]
            print(write_list)
            os.chdir('C:\Users\Ren\workspace2\DisHD\output')
            with open('file-'+str(file_id)+'-svm.csv','a') as fid:
                fid_csv = csv.writer(fid)
                fid_csv.writerows(write_list)
            
            
            gc.collect()
