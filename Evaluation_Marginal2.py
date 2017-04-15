import Get_Params
import Get_Rappor
import numpy
import itertools
import random
import time
import os
import csv


from JunctionTree import independent_marginal2,independent_marginal,independent_marginal3
from Estimate_Joint_Distribution import true_joint_distribution, unfold_pro_list
from numpy import power


def get_clique(range_size,clique_size,sample_size):
     ini_list2=list(itertools.combinations(range(range_size),clique_size))
     zzz=[list(eachtuple) for eachtuple in ini_list2]
     random.seed(15)
     zlist=random.sample(zzz,sample_size)
     return zlist
 
def l2_err(pro,true_pro):
     leng=len(pro)
     delta_pro=numpy.array(pro)-numpy.array(true_pro)
     return 1.0*numpy.sqrt(numpy.sum(numpy.power(delta_pro,2))/(1.0))
 
def get_avd(pro,true_pro):
    leng=len(pro)
    delta_pro=numpy.array(pro)-numpy.array(true_pro)
#     max_delta=max(numpy.abs(delta_pro))
#     return max_delta
    abs_delta=numpy.abs(delta_pro)
    return numpy.sum(abs_delta)/2.0
 
def cos(vector1,vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5) 
    
def get_max(pro,true_pro):
    max1=max(pro)
    max2=max(true_pro)
    
    return abs(max1-max2)

def get_var(pro,true_pro):
     return numpy.var(numpy.array(pro)-numpy.array(true_pro))

fai_C=0.4    #from 0.2, 0.3, 0.4, 0.5
f=0.5  # from 0.1, 0.2, 0.3, 0.4, 0.5  *********
bloombit=32
hashbit=16
dt=0.01
readlimit=60000
samplerate=1
sparse_rate=0.0
for file_id in [4]:
    
    #att_num1,node_num1,true_node_num1,rowlist1,multilist1=Get_Params.get_file_info(file_id,readlimit,1.0)
    if samplerate!=2 :
        att_num2,node_num2,true_node_num2,rowlist2,multilist2=Get_Params.get_file_info(file_id,readlimit,samplerate)
        
        att1_clique=get_clique(att_num2,1,10)
        att2_clique=get_clique(att_num2, 2, 20)   #[[0,1],[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[2,5],[2,7],[7,10]]
        att3_clique=get_clique(att_num2,3,20)  #[[0,1,2],[6,7,8],[9,10,11],[12,13,14],[3,5,7],[9,11,13],[2,13,15],[3,5,9],[4,5,15],[8,11,13],[9,10,15]]
        att4_clique=get_clique(att_num2,4,20)     #[[0,1,2,3],[4,5,6,7],[8,9,10,11],[2,4,6,8],[8,10,12,14],[3,6,8,11],[4,6,7,9],[2,4,7,8],[5,8,9,11],[3,5,6,14]]
        att6_clique=get_clique(att_num2,6,20)   #[[0,1,2,3,4,5],[6,7,8,9,10,11]]
        att8_clique=get_clique(att_num2,8,100)    #[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]
        att7_clique=get_clique(att_num2,7,100)
        att5_clique=get_clique(att_num2,5,20)
            
        if file_id==4:
            bloombit=32
            hashbit=4
            samplerate=0.1
            cluster_list=[att3_clique,att5_clique]
        else:
            bloombit=128
            hashbit=4
            samplerate=0.02
            cluster_list=[att2_clique,att3_clique]
        
        #for f in [0.98,0.97,0.96]:
        for f in [0.1,0.3,0.5,0.7,0.9,0.7,0.75,0.82,0.88,0.94]:
        #for f in [0.845,0.922,0.984,0.992,0.998]: #epsilon=10,5,1,0.5,0.1
        #for dpepsilon in [5,4,3,2,1]:
            #f=2/(numpy.exp(dpepsilon/(2.0*hashbit))+1)
            #print('f is: ', f)

            print('file_id',file_id,'samplerate:',samplerate,'sparse_rate:',sparse_rate,'f',f)

            att_num2, node_num2,true_node_num2,rowlist_sparse,multilist_sparse,bit_cand_list3,bit_list3,bitsum_list3=Get_Rappor.Get_rid_sparse(file_id,readlimit,samplerate,bloombit,hashbit,f,sparse_rate)
            #att_num22, node_num22,true_node_num22,rowlist_sparse2,multilist_sparse2,bit_cand_list32,bit_list32,bitsum_list32=Get_Rappor.Get_rid_sparse(file_id,readlimit,1,bloombit,hashbit,f,sparse_rate)
            #print(bitsum_list3)
#            bit_cand_list3,bit_list3,bitsum_list3=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num2,node_num2,true_node_num2,rowlist_sparse,multilist_sparse)
        
            freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num2, node_num2, rowlist_sparse, multilist_sparse)
            #print('finish basis!')
            #print(multilist_sparse)
            
#             att_clique=range(att_num2)
#             ini_list2=list(itertools.combinations([1,2,3,4],2))
#             z=[list(eachtuple) for eachtuple in ini_list2]
            
            #att2_clique=[[0,1],[1,2],[0,2]]
            
            #print(att2_clique)
            #print(att3_clique)
            #print(att4_clique)
            #print(att6_clique)
            #print(att8_clique)
            
            if file_id!=5:
                           
                for each_k in cluster_list:
                    lenk=len(each_k[0])
                    sum_err1=0.0
                    sum_err2=0.0
                    sum_err3=0.0
                    i=0
                    etime1=0.0
                    etime2=0.0
                    etime3=0.0
                    err_list1=[]
                    err_list2=[]
                    err_list3=[]
                    for eachclique in each_k:
                        true_list,true_pro=true_joint_distribution(multilist2, rowlist2, eachclique)
                        #print('true:',true_pro)
                        curr_time1=time.time()
                        
                        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2,bitsum_list3, f, dt)
                        #some_list=true_list
                        #pro1=true_pro
                        #some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist_sparse, bitsum_list3, f, dt)
                        #elapse=time.time()-curr_time
                        #print('esti1:',pro1,etime1)
                        curr_time2=time.time()
                        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                        curr_time3=time.time()
                        some_list,pro3=independent_marginal3(eachclique, bit_list3, bit_cand_list3, rowlist2,bitsum_list3, f, dt)
                        curr_time4=time.time()
                        etime1+=curr_time2-curr_time1
                        etime2+=curr_time3-curr_time2
                        etime3+=curr_time4-curr_time3

                        #print('esti1:',pro1)
                        #print('esti2:',pro2)
                        i+=1
                
                        err1=get_avd(pro1, true_pro)
                        err2=get_avd(pro2, true_pro)
                        err3=get_avd(pro3, true_pro)
                        sum_err1+=err1
                        sum_err2+=err2
                        sum_err3+=err3
                        #if (i%10<=10):
                        #err_list1.append(err1)
                        #err_list2.append(err2)
                        print(i,err1,err2,err3,etime1,etime2,etime3,eachclique)
                        #print(err_list1)
                        
                    mean_err1=1.0*sum_err1/len(each_k)
                    mean_err2=1.0*sum_err2/len(each_k)
                    mean_err3=1.0*sum_err3/len(each_k)
                    mean_time1=1.0*etime1/len(each_k)
                    mean_time2=1.0*etime2/len(each_k)
                    mean_time3=1.0*etime3/len(each_k)
                    print(lenk,mean_err1,mean_err2,mean_err3,mean_time1,mean_time2,mean_time3)
                    write_list=[[f,samplerate,lenk,mean_err1,mean_err2,mean_err3,mean_time1,mean_time2,mean_time3,sparse_rate,bloombit,hashbit]]
                    print(write_list)
                    os.chdir('C:\Users\Ren\workspace2\DisHD\output')
                    with open('file-'+str(file_id)+'-marginal.csv','a') as fid:
                        fid_csv = csv.writer(fid)
                        fid_csv.writerows(write_list)
#                     with open('file-'+str(file_id)+'-f-'+str(f)+'-errorlist.csv','a') as fid:
#                             fid_csv = csv.writer(fid)
#                             fid_csv.writerows([err_list1])
#                             fid_csv.writerows([err_list2])
                            
                #exit(0)    

                
                
                

'''
                mean_err=0
                i=0
                for eachclique in att3_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att3_clique)
                print('3-way',mean_err)
                 
                mean_err=0
                i=0
                for eachclique in att4_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att4_clique)
                print('4-way',mean_err)
                 
                 
                mean_err=0
                i=0
                for eachclique in att5_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att5_clique)
                print('5-way',mean_err)
                
                
                mean_err=0
                i=0
                for eachclique in att6_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att6_clique)
                print('6-way',mean_err)
                
                
                mean_err=0
                i=0
                for eachclique in att7_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att7_clique)
                print('7-way',mean_err)
                
                
                mean_err=0
                i=0
                for eachclique in att8_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att8_clique)
                print('8-way',mean_err)
                
                
                
                
            else:
                mean_err=0
                mean_time=0
                i=0
                for eachclique in att2_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    #some_list,pro=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
#                     print('true:',true_pro)
#                     print('esti2:',pro)
                    i+=1
            
                    err=get_avd(pro, true_pro)
                    mean_err+=err
                    mean_time+=elapse
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att2_clique)
                mean_time=mean_time/len(att2_clique)
                print('2-way',mean_err,mean_time)
                
                mean_err=0
                mean_time=0
                i=0
                for eachclique in att3_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    #some_list,pro=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
                    elapse=time.time()-curr_time
                       
                    true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
            #         print('true:',true_pro)
            #         print('esti2:',pro)
                    i+=1
              
                    err=get_avd(pro, true_pro)
                    mean_err+=err
                    mean_time+=elapse
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att3_clique)
                mean_time=mean_time/len(att3_clique)
                print('3-way',mean_err,mean_time)
                 
                  
                mean_err=0
                mean_time=0
                i=0
                for eachclique in att4_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                       
                    true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
            #         print('true:',true_pro)
            #         print('esti2:',pro)
                    i+=1
              
                    err=get_avd(pro, true_pro)
                    mean_err+=err
                    mean_time+=elapse
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att4_clique)
                mean_time=mean_time/len(att4_clique)
                print('4-way',mean_err,mean_time)
             
#             
#             
#             
#                 mean_err=0
#                 mean_time=0
#                 i=0
#                 for eachclique in att5_clique:
#                     curr_time=time.time()
#                     some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
#                     elapse=time.time()-curr_time
#                      
#                     true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
#             #         print('true:',true_pro)
#             #         print('esti2:',pro)
#                     i+=1
#             
#                     err=get_avd(pro, true_pro)
#                     mean_err+=err
#                     mean_time+=elapse
#                     #print(i,err,elapse)
#                 mean_err=1.0*mean_err/len(att5_clique)
#                 mean_time=mean_time/len(att5_clique)
#                 print('5-way',mean_err,mean_time)
#                 
                
                
#                 mean_err=0
#                 mean_time=0
#                 i=0
#                 for eachclique in att6_clique:
#                     curr_time=time.time()
#                     some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
#                     elapse=time.time()-curr_time
#                      
#                     true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
#             #         print('true:',true_pro)
#             #         print('esti2:',pro)
#                     i+=1
#             
#                     err=get_avd(pro, true_pro)
#                     mean_err+=err
#                     mean_time+=elapse
#                     #print(i,err,elapse)
#                 mean_err=1.0*mean_err/len(att6_clique)
#                 mean_time=mean_time/len(att6_clique)
#                 print('6-way',mean_err,mean_time)
#                 
                
                
#                 mean_err=0
#                 mean_time=0
#                 i=0
#                 for eachclique in att7_clique:
#                     curr_time=time.time()
#                     some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
#                     elapse=time.time()-curr_time
#                      
#                     true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
#             #         print('true:',true_pro)
#             #         print('esti2:',pro)
#                     i+=1
#             
#                     err=get_avd(pro, true_pro)
#                     mean_err+=err
#                     mean_time+=elapse
#                     #print(i,err,elapse)
#                 mean_err=1.0*mean_err/len(att7_clique)
#                 mean_time=mean_time/len(att7_clique)
#                 print('7-way',mean_err,mean_time)
#                 
#                 
#                 
#                 mean_err=0
#                 mean_time=0
#                 i=0
#                 for eachclique in att8_clique:
#                     curr_time=time.time()
#                     some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
#                     elapse=time.time()-curr_time
#                      
#                     true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
#             #         print('true:',true_pro)
#             #         print('esti2:',pro)
#                     i+=1
#             
#                     err=get_avd(pro, true_pro)
#                     mean_err+=err
#                     mean_time+=elapse
#                     #print(i,err,elapse)
#                 mean_err=1.0*mean_err/len(att8_clique)
#                 mean_time=mean_time/len(att8_clique)
#                 print('8-way',mean_err,mean_time)
'''                
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
    