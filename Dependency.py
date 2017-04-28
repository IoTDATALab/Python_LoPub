import Get_Params
import Get_Rappor
import Estimate_Joint_Distribution
import numpy
import scipy
import copy
import warnings
import csv
from numpy import reshape
warnings.filterwarnings("ignore")
#from scikits.statsmodels.tsa.ar_model import theta

def Get_Ent(attA_prob_list):
    Ent=0.0
    for i in range(len(attA_prob_list)):
        if (attA_prob_list[i]<=0):
            Ent+=0.0
        else:
            Ent+=attA_prob_list[i]*scipy.log(attA_prob_list[i])
    return -Ent

def Get_Rel_Ent(attA_prob_list):
    leng=len(attA_prob_list)
    total_Ent=-scipy.log(1.0/leng)
    Rel_Ent=(Get_Ent(attA_prob_list))/total_Ent
    
    return Rel_Ent

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

#print(Get_Rel_Ent([0.4,0.2,0.4]))

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
                #MI+=attAB_prob_list[ii]*scipy.log(1.0/(attA_prob_list[i]*attB_prob_list[j]))
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
    nonBinary=True
    RecordMi=[[0.0 for i in range(att_num)] for j in range(att_num)]
    Recordlist=[]
    if nonBinary:
        DepenGraph=[[0 for i in range(att_num)]for j in range(att_num)]
        CorrMatrix=[[0.0 for i in range(att_num)]for j in range(att_num)]
        #Rel_Ent=[Get_Ent(freqrate1[i])/(len(rowlist[i])-1) for i in range(att_num)]
        Rel_Ent=[Get_Ent(freqrate1[i]) for i in range(att_num)]
        RE_index=sorted(range(len(Rel_Ent)), key=lambda k: Rel_Ent[k])
        RE_index.reverse()
        leng_th=int(round(len(Rel_Ent)*(1)))
        print(Rel_Ent,RE_index)
        complex_time=0
        for ii in range(leng_th):
            #print(i,Get_Rel_Ent(freqrate1[i]))
            for jj in range(ii+1,leng_th):
                complex_time+=1
                i=RE_index[ii]
                j=RE_index[jj]
                if i>j:
                    temp=copy.copy(i)
                    i=copy.copy(j)
                    j=copy.copy(temp)
                    #print('temp',i,j,temp)
                kk=0
                for iii in range(att_num):
                    for jjj in range(iii+1,att_num):
                        kk=kk+1
                        #print(iii,jjj,kk)
                        if (iii==i) and (jjj==j):
                            k=kk-1
                #print('i,j,k:',i,j,k)       
                theta=min(len(freqrate1[i])-1,len(freqrate1[j])-1)*(fai_C**2)/2.0
                #print(freqrate1[i])
                #print(freqrate1[j])
                #print(freqrate2[k])
                True_Mi=True_MI(freqrate1[i], freqrate1[j], freqrate2[k])
                CorrMatrix[ii][jj]=True_Mi
                CorrMatrix[jj][ii]=True_Mi
                Recordlist.append(True_Mi)
                print('MI:',i,j,theta,True_Mi/min(len(rowlist[i]),len(rowlist[j])),Get_Ent(freqrate1[i])/len(rowlist[i]),Get_Ent(freqrate1[j])/len(rowlist[j]))
                #print('MI:',i,j,theta,True_Mi,Get_Ent(freqrate1[i]),Get_Ent(freqrate1[j]))
                
                k+=1
                if True_Mi>theta:
                    DepenGraph[i][j]=1
                    DepenGraph[j][i]=1
        ns=[[1 for i in range(att_num)]]
    else:
        DepenGraph=[[1 for i in range(att_num)]for j in range(att_num)]
        CorrMatrix=[[0.0 for i in range(att_num)]for j in range(att_num)]
        Rel_Ent=[Get_Ent(freqrate1[i])/(len(rowlist[i])-1) for i in range(att_num)]
        RE_index=sorted(range(len(Rel_Ent)), key=lambda k: Rel_Ent[k])
        #RE_index.reverse()
        leng_th=int(round(len(Rel_Ent)*(1-fai_C)))
        print(Rel_Ent,RE_index)
        complex_time=0
        for ii in range(leng_th):
            #print(i,Get_Rel_Ent(freqrate1[i]))
            for jj in range(ii+1,leng_th):
                complex_time+=1
                i=RE_index[ii]
                j=RE_index[jj]
                if i>j:
                    temp=copy.copy(i)
                    i=copy.copy(j)
                    j=copy.copy(temp)
                    #print('temp',i,j,temp)
                kk=0
                for iii in range(att_num):
                    for jjj in range(iii+1,att_num):
                        kk=kk+1
                        #print(iii,jjj,kk)
                        if (iii==i) and (jjj==j):
                            k=kk-1
                #print('i,j,k:',i,j,k)       
                theta=min(len(freqrate1[i])-1,len(freqrate1[j])-1)*(fai_C**2)/2.0
                #print(freqrate1[i])
                #print(freqrate1[j])
                #print(freqrate2[k])
                True_Mi=True_MI(freqrate1[i], freqrate1[j], freqrate2[k])
                CorrMatrix[i][j]=True_Mi
                CorrMatrix[j][i]=True_Mi
                #Recordlist.append(CorrMatrix[i][j])
                #print('MI:',i,j,theta,True_Mi,Get_Ent(freqrate1[i])/len(rowlist[i]),Get_Ent(freqrate1[j])/len(rowlist[j]))
                #print('MI:',i,j,theta,True_Mi,Get_Ent(freqrate1[i]),Get_Ent(freqrate1[j]))
                k+=1
                if True_Mi<theta:
                    DepenGraph[i][j]=0
                    DepenGraph[j][i]=0
        ns=[[1 for i in range(att_num)]]
    
    DepenGraph_true=[[0 for i in range(att_num)]for j in range(att_num)]
    CorrMatrix_true=[[0.0 for i in range(att_num)]for j in range(att_num)]
    
    k=0
    for i in range(att_num):
        #print(i,Get_Rel_Ent(freqrate1[i]))
        for j in range(i+1,att_num):

            
     
            theta=min(len(freqrate1[i])-1,len(freqrate1[j])-1)*(fai_C**2)/2.0
            #print(freqrate1[i])
            #print(freqrate1[j])
            #print(freqrate2[k])
            True_Mi=True_MI(freqrate1[i], freqrate1[j], freqrate2[k])
            CorrMatrix_true[i][j]=True_Mi
            CorrMatrix_true[j][i]=True_Mi
            #print('MI:',i,j,theta,True_Mi,Get_Ent(freqrate1[i])/len(rowlist[i]),Get_Ent(freqrate1[j])/len(rowlist[j]))
            #print('MI:',i,j,theta,True_Mi,Get_Ent(freqrate1[i]),Get_Ent(freqrate1[j]))
            k+=1
            if True_Mi>theta:
                DepenGraph_true[i][j]=1
                DepenGraph_true[j][i]=1
    #ns=[[1 for i in range(att_num)]]
    print('lost ratio:',1.0*sum(reshape(numpy.array(DepenGraph_true)-numpy.array(DepenGraph),att_num*att_num))/sum(reshape(numpy.array(DepenGraph_true),att_num*att_num)),sum(reshape(numpy.array(DepenGraph),att_num*att_num)),sum(reshape(numpy.array(DepenGraph_true),att_num*att_num)))
    #print('lost ratio:',1.0*sum(reshape(numpy.array(DepenGraph_true)-numpy.array(DepenGraph),att_num*att_num))/(att_num*att_num))
    print('complexity reduction:',1.0*(att_num*(att_num-1)/2-complex_time)/(att_num*(att_num-1)/2),complex_time,(att_num*(att_num-1)/2))
#     for i in range(att_num):
#         for j in range(att_num):
#             RecordMi[i][j]=CorrMatrix[i][j]/min(len(rowlist[i]),len(rowlist[j]))
#             Recordlist.append(RecordMi[i][j])
        
    #recordlist.append([ii,jj,CorrMatrix[i][j]/min(len(rowlist[i]),len(rowlist[j]))])
    print(Recordlist)
#     with open('mirecord.csv','wb') as fid:
#                     fid_csv = csv.writer(fid)
#                     fid_csv.writerows(Recordlist)
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
    
    Rel_Ent=[Get_Ent(p_single[i])/(len(rowlist[i])-1) for i in range(att_num)]
    RE_index=sorted(range(len(Rel_Ent)), key=lambda k: Rel_Ent[k])
    RE_index.reverse()
    leng_th=int(len(Rel_Ent)*(1-fai_C))
    print(Rel_Ent)
    print(RE_index)
    
    print('p_single via lasso:',p_single)
#     for i in range(att_num):   
#         print(i,Get_Rel_Ent(p_single[i]))
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
    for ii in range(leng_th):   
        #print(i,Get_Rel_Ent(p_single[i]))    
        for jj in range(ii+1, leng_th):
            i=RE_index[ii]
            j=RE_index[jj]
            theta=min(len(bit_cand_list[i])-1,len(bit_cand_list[j])-1)*(fai_C**2)/2.0
            p_comb2=(Estimate_Joint_Distribution.estimate_2d(bit_list[i], bit_list[j], bit_cand_list[i], bit_cand_list[j],bitsum_list,[i,j], f, 0.001))
            #p_comb2=Estimate_Joint_Distribution.estimate_2d2(bit_cand_list[i], bit_cand_list[j], bitsum_list, [i,j])
            p_comb=p_comb2
            #if (j!=25):
                #print('p comb:',p_comb)
            #    print('p_comb2:',i,j,p_comb2)
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
            print('Computing the mutual information:',i,j,Mi,Mi2,theta)
        p_single_list.append(p_single)
    ns=[[1 for ii in range(att_num)]]
    return CorrMatrix,DepenGraph,ns,att_num,node_num,origin_node_num,rowlist,multilist,bit_cand_list,bit_list,bitsum_list,p_comb_list,p_single_list


#DepenGraph=Get_Dep_Graph(4, 0.2, 0.01)
#a=[0.7,0.3]
#b=[0.6,0.4]
#c=[0.45,0.15,0.25,0.15]
#c=[[0.45,0.15],[0.25,0.15]]
#d=Get_MI(a, b, c)
#print(d)   