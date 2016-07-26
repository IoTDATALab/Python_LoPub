# import numpy
# import graph
import random
# import csv
#import Get_Params
#import Get_Rappor
#import Dependency
#import os
#import copy
#import time
from copy import copy
from Estimate_Joint_Distribution import att_combin, estimate_2d,\
    list_product,row_product, rappor_list_paste, true_joint_distribution, estimate_2d2,\
    row_list_product
import Get_Rappor


def independent_random_pick(some_list,probabilities):
    ################################################
    #To randomly (with proability in probabilities) pick an item from the some_list, this is independent
    #print(some_list,probabilities)
    x=random.uniform(0,1)
    cumulative_probability=0.0
    loc_some_list=list(range(len(some_list)))
    loc_each=0
    for loc_each,loc_probability in zip(loc_some_list,probabilities):
        cumulative_probability+=loc_probability
        if x < cumulative_probability: 
            break
    #print('loc each',loc_each,'some_list',some_list)
    #print(some_list)
    if len(some_list[0])==1:
        return [some_list[loc_each]]
    else:
        return some_list[loc_each]
#print(independent_random_pick([1,2,3,4,5,6], [0.0,0.0,0.0,0.0,0.1,0.9]))
#exit(0)

def conditional_random_pick(new_data,condition_list,condition_some_list,some_list,probabilities):
    #############################################################################################
    #To randomly pick an item from the some_list, according to the available_data
    #condition_set is the set of the conditions, condition_list is the list of possible conditions
    #some_list is the acceptable items
    condition_data=simple_conditiondata_combin(new_data, condition_list)
    #print(new_data,condition_data)
    con_loc=condition_some_list.index(condition_data)
    #print('conloc',con_loc,probabilities[con_loc])
    con_probabilities=probabilities[con_loc]
    x=random.uniform(0,1)
    cumulative_probability=0.0
    #loc_specify=0
    loc_some_list=list(range(len(some_list)))
    for loc_each,loc_probability in zip(loc_some_list,con_probabilities):
        cumulative_probability+=loc_probability
        if x < cumulative_probability: 
            #loc_specify=loc
            break
    if len(some_list[0])==1:
        return [some_list[loc_each]]
    else:
        return some_list[loc_each]

def cliques_to_locs(cliques_list):
    loc_list=[]
    for cliq in cliques_list:
        loc_list.append([ int(i) for i in cliq])
    return loc_list

def simple_conditiondata_combin(data_list, loc_list):
    if len(loc_list)==1:
        data_list_combine=data_list[loc_list[0]]
    else:
        
        data_list_combine=[]
        for loc in loc_list:
            data_list_combine.append(data_list[loc])
    return data_list_combine

#def pairwise_independent_margin(first_att_index,second_att_index,p_comb_list,row_list):

def independent_marginal(clique,bit_list,bit_cand_list,row_list,f,dt):   
    ##################################################################
    # To generate independent probability and possible list for sampling
    leng=len(clique)
    pro=[]
    some_list=[]
    if leng<=2:
        att_index1=clique[0]
        att_index2=clique[leng-1]
        proe=estimate_2d(bit_list[att_index1], bit_list[att_index2], bit_cand_list[att_index1], bit_cand_list[att_index2], f, dt)
        proleng=len(proe)
        for i in range(proleng):    
            pro.extend(proe[i])
        some_list=row_list[clique[0]]
        
    else:
        att_index1=clique[0]
        att_indexs=clique[1:(leng)]
        #print(att_index1)
        #print(att_indexs)
        att_rappor_list_combine,att_signal_list_combine,att_row_list_combine=att_combin(bit_list,bit_cand_list,row_list,att_indexs)
        #print('rowlist combine',att_row_list_combine)
        #print(att_row_list_combine)
        #proe=estimate_2d2(bit_cand_list[att_index1],att_signal_list_combine,bitsum_list,clique)
        proe=estimate_2d(bit_list[att_index1],att_rappor_list_combine,bit_cand_list[att_index1],att_signal_list_combine,f,dt)
        proleng=len(proe)
        for i in range(proleng):    
            pro.extend(proe[i])
        some_list=list_product(row_list[clique[0]],att_row_list_combine)
        #print('estimate:',proe)
    return some_list, pro
           
def independent_marginal2(clique,bit_list,bit_cand_list,row_list,bitsum_list,f,dt):   
    ##################################################################
    # To generate independent probability and possible list for sampling
    #print(row_list)
    leng=len(clique)
    #print(leng)
    pro=[]
    some_list=[]
    if leng==1:
        att_index1=clique[0]
        ptemp=Get_Rappor.lasso_regression([bit_cand_list[att_index1]], [bitsum_list[att_index1]])
        pro=ptemp[0]
        
        some_list=row_list[clique[0]]
        
    if leng==2:
        att_index1=clique[0]
        att_index2=clique[leng-1]
        ptemp=estimate_2d2(bit_cand_list[att_index1],bit_cand_list[att_index2],bitsum_list,clique)
        #Get_Rappor.lasso_regression([bit_cand_list[att_index1]], [bitsum_list[att_index1]])
        proe=ptemp
        proleng=len(proe)
        for i in range(proleng):    
            pro.extend(proe[i])
        #print('lasso estimate:',pro)
        #some_list=list_product([row_list[clique[0]]],[row_list[clique[1]]])
        some_list=row_product(row_list[clique[0]],row_list[clique[1]])
        
    if leng>2:
        att_index1=clique[0]
        att_indexs=clique[1:(leng)]
        #print(att_index1)
        #print(att_indexs)
        att_rappor_list_combine,att_signal_list_combine,att_row_list_combine=att_combin(bit_list,bit_cand_list,row_list,att_indexs)
        #print('rowlist combine',att_row_list_combine)
        #print(att_row_list_combine)
        proe=estimate_2d2(bit_cand_list[att_index1],att_signal_list_combine,bitsum_list,clique)
        #print(proe)
        #pro=proe[0]
        #proe=estimate_2d(bit_list[att_index1],att_rappor_list_combine,bit_cand_list[att_index1],att_signal_list_combine,f,dt)
        proleng=len(proe)
        for i in range(proleng):    
            pro.extend(proe[i]) 
        #some_list=list_product(row_list[clique[0]],att_row_list_combine)
        some_list=row_product(row_list[clique[0]],att_row_list_combine)
        #print('lasso estimate:',pro)
    some_list=row_list_product(row_list,clique)
    #print(some_list)
    #print(pro)
    return some_list, pro

def conditional_marginal(condition_list,clique,bit_list,bit_cand_list,row_list,f,dt):   
    ####################################################################################
    # To generate conditional probability and acceptable items for sampling
    pro=[]
    some_list=[]
    rest_list=list(set(clique)-set(condition_list))
    rest_list.sort()
    
    if len(condition_list)==1:
        att_indexs1=condition_list[0]
        att_rappor_list_combine1=bit_list[att_indexs1]
        att_signal_list_combine1=bit_cand_list[att_indexs1]
        att_row_list_combine1=row_list[att_indexs1] 
    else:
        att_indexs1=condition_list
        att_rappor_list_combine1,att_signal_list_combine1,att_row_list_combine1=att_combin(bit_list,bit_cand_list,row_list,att_indexs1)
        
    if len(rest_list)==1:
        att_indexs2=rest_list[0]
        att_rappor_list_combine2=bit_list[att_indexs2]
        att_signal_list_combine2=bit_cand_list[att_indexs2]
        att_row_list_combine2=row_list[att_indexs2]
    else:
        att_indexs2=rest_list
        att_rappor_list_combine2,att_signal_list_combine2,att_row_list_combine2=att_combin(bit_list,bit_cand_list,row_list,att_indexs2)
    
    
    pro=estimate_2d(att_rappor_list_combine1,att_rappor_list_combine2,att_signal_list_combine1,att_signal_list_combine2,f,dt)
    
    condition_some_list=att_row_list_combine1
    some_list=att_row_list_combine2
    
    #print(condition_some_list)
    #print(some_list)
    #print('true:',true_joint_distribution(bit_list,row_list,clique))
    print('estimate:',pro)
 
    return condition_some_list,some_list, pro 

def conditional_marginal2(condition_list,clique,bit_list,bit_cand_list,row_list,bitsum_list,f,dt):   
    ####################################################################################
    # To generate conditional probability and acceptable items for sampling
    pro=[]
    some_list=[]
    rest_list=list(set(clique)-set(condition_list))
    rest_list.sort()
    
    if len(condition_list)==1:
        att_indexs1=condition_list[0]
        att_rappor_list_combine1=bit_list[att_indexs1]
        att_signal_list_combine1=bit_cand_list[att_indexs1]
        att_row_list_combine1=row_list[att_indexs1]
    else:
        att_indexs1=condition_list
        att_rappor_list_combine1,att_signal_list_combine1,att_row_list_combine1=att_combin(bit_list,bit_cand_list,row_list,att_indexs1)
        
    if len(rest_list)==1:
        att_indexs2=rest_list[0]
        att_rappor_list_combine2=bit_list[att_indexs2]
        att_signal_list_combine2=bit_cand_list[att_indexs2]
        att_row_list_combine2=[row_list[att_indexs2]]
    else:
        att_indexs2=rest_list
        att_rappor_list_combine2,att_signal_list_combine2,att_row_list_combine2=att_combin(bit_list,bit_cand_list,row_list,att_indexs2)
    
    
    proe=estimate_2d2(att_signal_list_combine1,att_signal_list_combine2,bitsum_list,clique)
    #print(proe)
    protemp=proe
    #lenpro=len(protemp)
    leng1=len(att_signal_list_combine1)
    leng2=len(att_signal_list_combine2)
    ptemp=[[]for i in range(leng1)]
    for i in range(leng1):
        for j in range(leng2):
            ptemp[i].append(protemp[i][j])
    condition_some_list=att_row_list_combine1
    #some_list=att_row_list_combine2
    some_list=row_list_product(row_list, rest_list)
    
    #print(condition_some_list)
    #print(some_list)
    #print('true:',true_joint_distribution(bit_list,row_list,clique))
    #print('estimate:',ptemp)
 
    return condition_some_list,some_list, ptemp


def pairwise_conditional_margin(first_att_index,second_att_index,p_comb_list,row_list):
    if first_att_index>second_att_index:
        temp_index=copy(second_att_index)
        second_att_index=copy(first_att_index)
        first_att_index=copy(temp_index)
    index=0
    for i in range(len(row_list)):
        for j in range(i+1,len(row_list)):
            index+=1
            if (i==first_att_index and j==second_att_index):
                index=index-1
                break
    print('index:',first_att_index,second_att_index,index,len(p_comb_list))
    pro=p_comb_list[index]
    condition_some_list=row_list[first_att_index]
    some_list=row_list[second_att_index]
    if first_att_index>second_att_index:
        pro=map(list,zip(*pro))
    return condition_some_list,some_list,pro

def pair_independe_draw(new_data_list,origin_node_num,clique,p_single_list,p_comb_list,row_list):
    leng=len(clique)
    if leng==1:
        some_list=row_list[clique[0]]
        pro=p_single_list[clique[0]]
        for i in range(origin_node_num):
            items=independent_random_pick(some_list, pro)
            new_data_list[i][clique[0]]=items
    else:
        initial_flag=0
        for i_clique in range(leng-1):
            j_clique=i_clique+1

            condition_list,some_list,pro=pairwise_conditional_margin(clique[i_clique], clique[j_clique], p_comb_list, row_list)
            #print('pro:',pro,some_list)
            if initial_flag==0:
                initial_flag=1
                some_list2=list_product(condition_list, some_list)
                pro2=pro[0]+pro[1]
                for i in range(origin_node_num):
                    item=independent_random_pick(some_list2, pro2)
                    new_data_list[i][clique[i_clique]]=item[0]
                    new_data_list[i][clique[j_clique]]=item[1]            
            else:
                for i in range(origin_node_num):
                    item=conditional_random_pick(new_data_list[i], [clique[i_clique]], row_list[clique[j_clique]], some_list, pro)
                    new_data_list[i][clique[j_clique]]=item[0]
            
            print('Drawn',clique[i_clique],clique[j_clique])
                   
    return new_data_list

def independe_draw(ISflag,new_data_list,origin_node_num,clique,bit_list,bit_cand_list,row_list,p_single_list,p_comb_list,f,dt):
    if ISflag==1:
        some_list, pro=independent_marginal(clique, bit_list, bit_cand_list, row_list, f, dt)
        #some_list, pro=independent_marginal2(clique, bit_list, bit_cand_list, row_list, f, dt)
        for i in range(origin_node_num):
            items=independent_random_pick(some_list, pro)
            for j in range(len(clique)):
                new_data_list[i][clique[j]]=items[j]
    else:
        new_data_list=pair_independe_draw(new_data_list,origin_node_num,clique,p_single_list,p_comb_list,row_list)
           
    return new_data_list

def independe_draw2(ISflag,new_data_list,origin_node_num,clique,bit_list,bitsum_list,bit_cand_list,row_list,p_single_list,p_comb_list,f,dt):
    if ISflag==1:
        #some_list, pro=independent_marginal(clique, bit_list, bit_cand_list, row_list, f, dt)
        some_list, pro=independent_marginal2(clique, bit_list, bit_cand_list, row_list, bitsum_list,f, dt)
        #print('some',some_list)
        for i in range(origin_node_num):
            items=independent_random_pick(some_list, pro)
#             if i==1:
#                 print('item',items)
            for j in range(len(clique)):
                new_data_list[i][clique[j]]=items[j]
                #print(items[j])
    else:
        new_data_list=pair_independe_draw(new_data_list,origin_node_num,clique,p_single_list,p_comb_list,row_list)
           
    return new_data_list
            

def pair_conditional_draw(new_data_list,origin_node_num,condition_list,clique,p_comb_list,row_list):
    some_clique=list(set(clique)-set(condition_list))
    some_clique.sort()
    leng=len(some_clique)
    first_index=condition_list[0]
    second_index=some_clique[0]
    #print(first_index,second_index,p_comb_list,row_list)
    #exit(0)
    condition_some_list,some_list,pro=pairwise_conditional_margin(first_index, second_index, p_comb_list, row_list)
#     print(condition_some_list)
#     print(some_list)
#     print(pro)
    for i in range(origin_node_num):
            item=conditional_random_pick(new_data_list[i], [first_index],condition_some_list,some_list, pro)
           # for j in range(len(clique)):
            new_data_list[i][second_index]=item
    
    if leng>1:
        for i_clique in range(leng):
            for j_clique in range(i_clique,leng):
                
                condition_some_list,some_list,pro=pairwise_conditional_margin(some_clique[i_clique], some_clique[j_clique], p_comb_list, row_list)
                for i in range(origin_node_num):
                    item=conditional_random_pick(new_data_list[i],[some_clique[i_clique]], condition_some_list, some_list, pro)
                    new_data_list[i][some_clique[j_clique]]=item
    return new_data_list
        
        
    
    

def conditional_draw(ISflag,new_data_list,origin_node_num,condition_list,clique,bit_list,bit_cand_list,row_list,p_single_list,p_comb_list,f,dt):
    if ISflag==1:
        some_clique=list(set(clique)-set(condition_list))
        some_clique.sort()
        condition_some_list,some_list, pro=conditional_marginal2(condition_list, clique, bit_list, bit_cand_list, row_list, f, dt)
        for i in range(origin_node_num):
            new_data=new_data_list[i]
            items=conditional_random_pick(new_data,condition_list, condition_some_list, some_list, pro)
            #print('condition pick', i, 'of',node_num)
            for j in range(len(set(clique)-set(condition_list))):
                new_data_list[i][some_clique[j]]=items[j]
            
    else:
        new_data_list=pair_conditional_draw(new_data_list,origin_node_num,condition_list,clique,p_comb_list,row_list)
            
    return new_data_list


def conditional_draw2(ISflag,new_data_list,origin_node_num,condition_list,clique,bit_list,bitsum_list,bit_cand_list,row_list,p_single_list,p_comb_list,f,dt):
    if ISflag==1:
        some_clique=list(set(clique)-set(condition_list))
        some_clique.sort()
        condition_some_list,some_list, pro=conditional_marginal2(condition_list, clique, bit_list, bit_cand_list, row_list, bitsum_list, f, dt)
        #print('cond',condition_list)
        #print('cosome',condition_some_list)
        #print('some',some_list)
        #print('pro:',pro)
        for i in range(origin_node_num):
            new_data=new_data_list[i]
            items=conditional_random_pick(new_data,condition_list, condition_some_list, some_list, pro)
            #print('condition pick', i, 'of',node_num)
#             if i in [1,2,3]:
#                 print('items',items)
            for j in range(len(set(clique)-set(condition_list))):
                new_data_list[i][some_clique[j]]=items[j]
                #print('ij',items[j])
            
    else:
        new_data_list=pair_conditional_draw(new_data_list,origin_node_num,condition_list,clique,p_comb_list,row_list)
            
    return new_data_list
######################################################################################################################################################################################################









##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################        
# ISflag=1
# file_id=2
# fai_C=0.45
# f=0.6
# bloombit=16
# hashbit=2
# dt=0.01
# readlimit=50000
# samplerate=0.01
# 
# param_string='D_'+str(file_id)+'_C_'+str(fai_C)+'_f_'+str(f)+'_B_'+str(bloombit)+'_H_'+str(hashbit)+'_S_'+str(samplerate)+'_R_'+str(readlimit)+'_T_'
# print(param_string)
# os.chdir('C:\Users\Ren\workspace2\DisHD\output')
# folder = param_string+time.strftime(r"%Y-%m-%d_%H-%M-%S",time.localtime())
# os.makedirs(r'%s/%s'%(os.getcwd(),folder))
# 
# att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(file_id,readlimit,samplerate)
# freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num, node_num, rowlist, multilist)
# bit_cand_list,bit_list,bitsum_list=Get_Rappor.rappor_process(file_id,readlimit,samplerate, bloombit, hashbit, f)
# 
# #TrueDepG,Truens,True_CorMat,freqrate2,freqrate1=Dependency.True_Dep_Graph(file_id, readlimit,samplerate, fai_C)
# TrueDepG,Truens,True_CorMat,freqrate2,freqrate1=Dependency.True_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,fai_C)
# TrueDG=numpy.array(TrueDepG)
# #print(True_CorMat)
# print(TrueDG)
# TrueNS=numpy.array(Truens)
# [True_jtree, True_root, True_cliques, True_B, True_w]=graph.graph_to_jtree(TrueDG,TrueNS)
# print(True_cliques)
# print(True_jtree)
# #print(freqrate2)
# 
# ########################################################################## Write Into Files#############################################################################
# 
# with open(folder+'\\'+'1-CorrMat_R.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(True_CorMat)
#     
# with open(folder+'\\'+'2-DGraph_R.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(TrueDepG)
#     
# with open(folder+'\\'+'3-JTree_R.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(True_jtree)
#     
# with open(folder+'\\'+'4-Cliques_R.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(True_cliques)
# 
# with open(folder+'\\'+'5-JDistribution2_R.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(freqrate2)
# 
# #print(freqrate2)
# with open(folder+'\\'+'5-JDistribution1_R.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(freqrate1)
# ##################################################################################################################################################################################################
# 
# 
# #Corr_Matrix,DepenGraph,ns,att_num,node_num,origin_node_num,row_list,multilist,bit_cand_list, bit_list,bitsum_list,p_comb_list,p_single_list=Dependency.Get_Dep_Graph(file_id,readlimit,samplerate,bloombit,hashbit,fai_C, f)
# Corr_Matrix,DepenGraph,ns,att_num,node_num,origin_node_num,row_list,multilist,bit_cand_list, bit_list,bitsum_list,p_comb_list,p_single_list=Dependency.Get_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,bloombit, hashbit, f,bit_cand_list,bit_list,bitsum_list,fai_C)
# DG=numpy.array(DepenGraph)
# print(DG)
# print(DG-TrueDG)
# ns=numpy.array(ns)
# [jtree, root, cliques, B, w]=graph.graph_to_jtree(DG,ns)
# 
# ############################################################ Write into files ############################################################################
# with open(folder+'\\'+'0-Infomation.txt','wb') as fid:
#     info_list=[['att_num',att_num],['node_num',node_num],['origin_node_num',origin_node_num],['sample_rate',samplerate],['file_id',file_id],['fai_C',fai_C],['f',f],['bloombit',bloombit],['hashbit',hashbit]]
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(info_list)
#     
# with open(folder+'\\'+'0-Domain.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(row_list)
#     
# with open(folder+'\\'+'0-DataSet.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(multilist)
# 
# 
# with open(folder+'\\'+'1-CorrMat_E.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(Corr_Matrix)
#    
# with open(folder+'\\'+'2-DGraph_E.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(DepenGraph)
#     
# with open(folder+'\\'+'3-JTree_E.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(jtree)
# 
# with open(folder+'\\'+'4-Cliques_E.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(cliques)
# 
# with open(folder+'\\'+'5-JDistribution1_E.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(p_single_list)
# 
# #print(p_comb_list)
# p_comb_len=len(p_comb_list)
# pp=[[]for i in range(p_comb_len)]
# for i in range(p_comb_len):
#     for p_sub in p_comb_list[i]:
#         pp[i].extend(p_sub)
# 
# with open(folder+'\\'+'5-JDistribution2_E.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(pp)
# 
# #exit(0)
# ##########################################################################################################################################################
# 
# 
# new_data_list=[['0' for i in range(att_num)]for j in range(origin_node_num)]
# ############################    EXAMPLE ####################################################################
# # MG=numpy.array([[0,1,0,0,0,0],[1,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,1],[0,0,1,1,0,0],[0,0,0,1,0,0]])
# #cliques=[[0, 1, 2, 6], [4, 6, 8, 10, 12, 14], [1, 2, 6, 7, 8, 12], [2, 5, 6, 7, 8, 12], [3, 5, 6, 7, 9, 15], [6, 7, 8, 10, 11, 12, 13, 14], [5, 6, 7, 8, 11, 12, 13, 14], [5, 6, 7, 8, 9, 11, 15], [5, 6, 7, 8, 11, 13, 14, 15]]
# # ns = numpy.array([[ 1, 1, 1,1,1,1]])
# # [jtree, root, cliques, B, w]=graph.graph_to_jtree(MG,ns)
# # print(B)
# # print(root)
# # print(cliques)
# # print(B)
# # print(w)
# #print(independent_random_pick(['a','b','c'], [0.2,0.5,0.3]))
# cliques_list=cliques_to_locs(cliques)
# 
# print(cliques_list)
# sampled_set=set()
# unsampl_list=copy(cliques_list)
# 
# while len(unsampl_list)>0 :
#     get_one=unsampl_list.pop()
#     print('computer new', get_one)
#     independe_draw(ISflag,new_data_list, origin_node_num, get_one, bit_list, bit_cand_list, row_list,p_single_list,p_comb_list, f, dt)
#     #print('true:',true_joint_distribution(multilist,row_list,get_one))
#     sampled_set=sampled_set|set(get_one)
#     visited_list=[]
#     while 1 :
#         for clique in unsampl_list:
#             condition_set=set(clique)&set(sampled_set)
#             visited_list.append(clique)
#             if len(condition_set)>0 :
#                 if  set(clique).issubset(sampled_set):
#                     print (clique,'already there!')
#                     unsampl_list.remove(clique)
#                     visited_list=[]
#                 else:
#                     unsampl_list.remove(clique)
#                     visited_list=[]
#                     condition_list=list(condition_set)
#                     condition_list.sort()
#                     print('computer condition', condition_list, clique)
#                     conditional_draw(ISflag,new_data_list, origin_node_num, condition_list, clique, bit_list, bit_cand_list, row_list,p_single_list,p_comb_list, f, dt)
#                     #print('true:',true_joint_distribution(multilist,row_list,clique))
#                     sampled_set=sampled_set|set(clique) 
#             visit=visited_list
#             unsample=unsampl_list
#             if visit==unsample:
#                 break
#         visit=visited_list
#         unsample=unsampl_list
#         if visit==unsample:
#             break
#         
# #print(new_data_list)
# 
# ############################################################ Write into files ############################################################################
# with open(folder+'\\'+'0-DataSyn.csv','wb') as fid:
#     fid_csv = csv.writer(fid)
#     fid_csv.writerows(new_data_list)
# ##########################################################################################################################################################
#     
# #import Evaluation_SVM
# #os.popen('python Evaluation_SVM.py')

























