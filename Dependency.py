import Get_Params
import Get_Rappor
import Estimate_Joint_Distribution
import numpy
import scipy
import warnings
warnings.filterwarnings("ignore")
#from scikits.statsmodels.tsa.ar_model import theta


def Get_MI(attA_prob_list,attB_prob_list,attAB_prob_list):
    MI=0.0
    for i in range(len(attA_prob_list)):
        for j in range(len(attB_prob_list)):
            if (attA_prob_list[i]*attB_prob_list[j])<=0 or attAB_prob_list[i][j]<=0  :
                MI+=0
            else:
                MI+=attAB_prob_list[i][j]*scipy.log(attAB_prob_list[i][j]/(attA_prob_list[i]*attB_prob_list[j]))
            #print(MI)
    return MI

def True_MI(attA_prob_list,attB_prob_list,attAB_prob_list):
    MI=0.0
    ii=0
    for i in range(len(attA_prob_list)):
        for j in range(len(attB_prob_list)):
            #MI=MI+attAB_prob_list[i*len(attB_prob_list)+j]*scipy.log(attAB_prob_list[i*len(attB_prob_list)+j]/(attA_prob_list[i]*attB_prob_list[j]))
            #print((i)*len(attB_prob_list)+j)
            if (attA_prob_list[i]*attB_prob_list[j])<=0 or attAB_prob_list[ii]<=0  :
                MI+=0
            else:
                MI+=attAB_prob_list[ii]*scipy.log(attAB_prob_list[ii]/(attA_prob_list[i]*attB_prob_list[j]))
            #print('comp mi:',attAB_prob_list[ii],attA_prob_list[i]*attB_prob_list[j],MI)
            ii+=1
            #print(attAB_prob_list[i][j])
    return MI
            
            
######################################################################################################################################################################################################    
# def True_Dep_Graph(file_id,readlimit,samplerate,fai_C):
#     
#     
#     att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(file_id,readlimit,samplerate)
#     freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num, node_num, rowlist, multilist)
def True_Dep_Graph(att_num,node_num,true_node_num,rowlist,multilist,freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist,fai_C):
    
    DepenGraph=[[0 for i in range(att_num)]for j in range(att_num)]
    CorrMatrix=[[0.0 for i in range(att_num)]for j in range(att_num)]
    ii=0
    for i in range(att_num):
        for j in range(i+1,att_num):
            theta=min(len(freqrate1[i])-1,len(freqrate1[j])-1)*(fai_C**2)/2.0
            #print(freqrate1[i])
            #print(freqrate1[j])
            #print(freqrate2[ii])
            True_Mi=True_MI(freqrate1[i], freqrate1[j], freqrate2[ii])
            CorrMatrix[i][j]=True_Mi
            CorrMatrix[j][i]=True_Mi
            #print('MI:',i,j,True_Mi)
            ii+=1
            if True_Mi>theta:
                DepenGraph[i][j]=1
                DepenGraph[j][i]=1
    ns=[[1 for i in range(att_num)]]
    return DepenGraph,ns,CorrMatrix,freqrate2,freqrate1
    

# def Get_Dep_Graph(file_id,readlimit,samplerate,num_bloom_bits,num_hash,fai_C, f):
# 
#     att_num,node_num,origin_node_num,rowlist,multilist=Get_Params.get_file_info(file_id,readlimit,samplerate)
#     bit_cand_list,bit_list,bitsum_list=Get_Rappor.rappor_process(file_id,readlimit,samplerate, num_bloom_bits, num_hash, f)
def Get_Dep_Graph(att_num,node_num,origin_node_num,rowlist,multilist,num_bloom_bits, num_hash, f,bit_cand_list,bit_list,bitsum_list,fai_C):
    Mi=0
    p_single=[]
    p_comb=[]
    p_comb_list=[]
    p_single_list=[]
    DepenGraph=[[0 for i in range(len(bit_cand_list))]for j in range(len(bit_cand_list))]
    CorrMatrix=[[0.0 for i in range(att_num)]for j in range(att_num)]
    
    
    p_single=Get_Rappor.lasso_regression(bit_cand_list, bitsum_list)
    print('p_single via lasso:',p_single)
#     for i in range(att_num):
#         p_temp=Estimate_Joint_Distribution.estimate_2d(bit_list[i], bit_list[i], bit_cand_list[i], bit_cand_list[i], f, 0.001)
#         p_temp_array=numpy.array(p_temp)
# #         p_freq=map(sum,p_temp)
# #         p_sum=sum(p_freq)
# #         p=[i/(p_sum+0.0) for i in p_freq]
#         
#         p=numpy.array([p_temp[0][0]+p_temp[1][0],p_temp[0][1]+p_temp[1][1]])/(sum(numpy.array([p_temp[0][0]+p_temp[1][0],p_temp[0][1]+p_temp[1][1]]))+0.0)
#         print('p_single:',p)
#         p_single.append(p)
#         #p_single.append([p_temp[0][0]+p_temp[1][0],p_temp[0][1]+p_temp[1][1]])
    for i in range(att_num):       
        for j in range(i+1, att_num):
            theta=min(len(bit_cand_list[i])-1,len(bit_cand_list[j])-1)*(fai_C**2)/2.0
            #p_comb=(Estimate_Joint_Distribution.estimate_2d(bit_list[i], bit_list[j], bit_cand_list[i], bit_cand_list[j], f, 0.001))
            p_comb2=Estimate_Joint_Distribution.estimate_2d2(bit_cand_list[i], bit_cand_list[j], bitsum_list, [i,j])
            p_comb=p_comb2
            if (i%10==0):
                #print('p comb:',p_comb)
                print('p_comb2:',i,j,p_comb2)
            p_comb_list.append(p_comb)
            
            #p_single1=[sum(eachlist) for eachlist in p_comb2]

            p_single1=[sum(eachlist) for eachlist in p_comb]
            p_comb_T=map(list,zip(*p_comb))
            p_single2=[sum(eachlist) for eachlist in p_comb_T]

            #print('p single1:',p_single1)
            #print('p single2:', p_single2)
            
            Mi=Get_MI(p_single1, p_single2, p_comb)
            Mi2=Get_MI(p_single[i], p_single[j], p_comb2)
            #Mi=Get_MI(p_single[i], p_single[j], p_comb)
            CorrMatrix[i][j]=Mi
            CorrMatrix[j][i]=Mi
            if Mi2>theta:
                DepenGraph[i][j]=1
                DepenGraph[j][i]=1
            #print('Computing the mutual information:',i,j,Mi,Mi2,theta)
        p_single_list.append(p_single1)
    ns=[[1 for i in range(att_num)]]
    return CorrMatrix,DepenGraph,ns,att_num,node_num,origin_node_num,rowlist,multilist,bit_cand_list,bit_list,bitsum_list,p_comb_list,p_single_list


#DepenGraph=Get_Dep_Graph(4, 0.2, 0.01)
#a=[0.7,0.3]
#b=[0.6,0.4]
#c=[0.45,0.15,0.25,0.15]
#c=[[0.45,0.15],[0.25,0.15]]
#d=Get_MI(a, b, c)
#print(d)   