import Get_Params
import numpy
import scipy
import copy
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model.coordinate_descent import ElasticNet
#from sklearn.linear_model.coordinate_descent import enet_path
#from Evaluation_Marginal import readlimit


def rappor_process(num_bloombits,num_hash,f,num_att,num_node,origin_node_num,list_att,list_data):

# def rappor_process(fid,readlimit,samplerate,num_bloombits,num_hash,f):
#     
# 
#     
#     num_att,num_node,origin_node_num,list_att,list_data=Get_Params.get_file_info(fid,readlimit,samplerate)
    freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(num_att, num_node, list_att, list_data)
    #e=Get_Params.set_rappor_params(num_bloombits, num_hash, f)
    
    bit_list=[[[0 for i in range(num_bloombits)]for j in range(num_att)] for k in range(num_node)]
    bit_cand_list=[[[0 for i in range(num_bloombits)]for j in range(len(list_att[k]))]for k in range(num_att)]
    bitsum_list=[[0 for i in range(num_bloombits)]for j in range(num_att)]
    
    
    for i in range(num_node):
        for j in range(num_att):
            #print(list_data[i][j],type(list_data[i][j]))
            e=Get_Params.set_rappor_params(num_bloombits, num_hash,num_att,f)
            bit_list[i][j]=Get_Params.get_B(list_data[i][j], e)
            #print(bit_list[i][j])
            bitsum_list[j]=list(map(lambda x: x[0]+x[1], zip(bitsum_list[j], bit_list[i][j])))
    
    bit_list=map(list, zip(*bit_list))
            
    #print('origin:',bitsum_list)
            
    for i in range(num_att):
        for j in range(len(list_att[i])):
            e=Get_Params.set_rappor_params(num_bloombits, num_hash,num_att,f)
            bit_cand_list[i][j]=Get_Params.get_S(list_att[i][j],e)
            
    tempbitsum_list=(numpy.array(bitsum_list)-(0.5*f)*num_node)/(1-f)
    #print(tempbitsum_list)
    bitsum_list=tempbitsum_list.tolist()
    
    print('f:',f,'sample B:',bit_list[0][0])
    print('f:',f,'sample S:',bit_cand_list[0][0])
            
    return bit_cand_list, bit_list, bitsum_list


#######################################################################################################################################################################################
def lasso_regression2(bit_cand_list,bitsum_list):
    #################
    ################### this function is used for all the data, no
    lasso_cf=[] 
    for i in range(len(bit_cand_list)):
        #print(bit_cand_list[i],bitsum_list[i])
        x=map(list, zip(*bit_cand_list[i]))
        y=bitsum_list[i]
        clf=LinearRegression()
        #clf = Lasso(alpha=1.0)
        #clf=ElasticNet(alpha=1.0, l1_ratio=2.0)
        m=clf.fit(x, y)
        
        coef=clf.coef_
        #print(coef)
        #index=coef.nonzero()
        ratio=coef/(sum(coef))
        lasso_cf.append(ratio.tolist())
        #print(clf.coef_)
         
    return lasso_cf
def lasso_regression(bit_cand_list,bitsum_list):
    #################
    ################### this function is used for all the data, no
    lasso_cf=[] 
    for i in range(len(bit_cand_list)):
        #print(bit_cand_list[i],bitsum_list[i])
        x=map(list, zip(*bit_cand_list[i]))
        y=bitsum_list[i]
        #clf=LinearRegression()
        clf = Lasso(alpha=1.0)
        #clf=ElasticNet(alpha=0.1, l1_ratio=2.0)
        m=clf.fit(x, y)
        coef=clf.coef_
        ratio=coef/(sum(coef))
        lasso_cf.append(ratio.tolist())
        #print(clf.coef_)
         
    return lasso_cf

def Get_rid_sparse(file_id,readlimit,samplerate,bloombit,hashbit,f,sparse_rate): 
#def Get_rid_sparse(bit_cand_list,bitsum_list,att_num,node_num,true_node_num,rowlist,multilist,sparse_rate): 
    
    att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(file_id,readlimit,samplerate)
    bit_cand_list,bit_list,bitsum_list=rappor_process(bloombit, hashbit, f,att_num,node_num,true_node_num,rowlist,multilist)
            
    
    p_single=lasso_regression(bit_cand_list, bitsum_list)
    #print(p_single)
    for i in range(att_num):
        lengi=len(p_single[i])
        rowcopy=copy.copy(rowlist[i])
        for j in range(lengi):
            if p_single[i][j]<sparse_rate:
                rowcopy.remove(rowlist[i][j])
                for k in range(node_num):
                    if multilist[k][i]==rowlist[i][j]:
                        multilist[k][i]='99'
        rowlist[i]=rowcopy
        if len(rowlist[i])<lengi:
            rowlist[i].append('99')
    bit_cand_list3,bit_list3,bitsum_list3=rappor_process(bloombit, hashbit, f,att_num,node_num,true_node_num,rowlist,multilist)
    return att_num,node_num,true_node_num,rowlist,multilist,bit_cand_list3,bit_list3,bitsum_list3
                      
########################################################################################################################################################################################    









