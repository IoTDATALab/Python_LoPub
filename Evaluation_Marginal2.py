import Get_Params
import Get_Rappor
import numpy
import itertools
import random
import time

from JunctionTree import independent_marginal2,independent_marginal
from Estimate_Joint_Distribution import true_joint_distribution, unfold_pro_list
from numpy import power

#file_id=2
for file_id in [4]:
    
    fai_C=0.4    #from 0.2, 0.3, 0.4, 0.5
    f=0.5  # from 0.1, 0.2, 0.3, 0.4, 0.5  *********
    bloombit=128
    hashbit=16
    dt=0.01
    readlimit=50000
    
    for samplerate in [0.0215]:
        #samplerate=0.0215  # from 0.01, 0.05, 0.1, 0.5, 1 0.0215
        
        def get_clique(range_size,clique_size,sample_size):
            ini_list2=list(itertools.combinations(range(range_size),clique_size))
            zzz=[list(eachtuple) for eachtuple in ini_list2]
            zlist=random.sample(zzz,sample_size)
            return zlist
        
        def l2_err(pro,true_pro):
            leng=len(pro)
            delta_pro=numpy.array(pro)-numpy.array(true_pro)
            return 1.0*numpy.sqrt(numpy.sum(numpy.power(delta_pro,2))/(1.0))
        
        def get_avd(pro,true_pro):
            leng=len(pro)
            delta_pro=numpy.array(pro)-numpy.array(true_pro)
            abs_delta=numpy.abs(delta_pro)
            return numpy.sum(abs_delta)/(2.0)
        
        def get_var(pro,true_pro):
            
            return numpy.var(numpy.array(pro)-numpy.array(true_pro))

    
        for sparse_rate in [0.10,0.05,0.01,0.00]:
            
            print('file_id',file_id,'samplerate:',samplerate,'sparse_rate:',sparse_rate)
                
            att_num1,node_num1,true_node_num1,rowlist1,multilist1=Get_Params.get_file_info(file_id,readlimit,1.0)
            att_num2,node_num2,true_node_num2,rowlist2,multilist2=Get_Params.get_file_info(file_id,readlimit,samplerate)
#             
#             bit_cand_list2,bit_list2,bitsum_list2=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num2,node_num2,true_node_num2,rowlist2,multilist2)
#             
            #sparse_rate=0.00
#            rowlist_sparse,multilist_sparse=Get_Rappor.Get_rid_sparse(bit_cand_list2,bitsum_list2,att_num2,node_num2,true_node_num2,rowlist2,multilist2,sparse_rate)
            att_num2, node_num2,true_node_num2,rowlist_sparse,multilist_sparse,bit_cand_list3,bit_list3,bitsum_list3=Get_Rappor.Get_rid_sparse(file_id,readlimit,samplerate,bloombit,hashbit,f,sparse_rate)
            #print(rowlist_sparse)
#            bit_cand_list3,bit_list3,bitsum_list3=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num2,node_num2,true_node_num2,rowlist_sparse,multilist_sparse)
            
            freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num2, node_num2, rowlist_sparse, multilist_sparse)
            print('finish basis!')
            #print(multilist_sparse)
            
            att_clique=range(att_num2)
            ini_list2=list(itertools.combinations([1,2,3,4],2))
            z=[list(eachtuple) for eachtuple in ini_list2]
            
            
            att2_clique=get_clique(15, 2, 100)   #[[0,1],[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[2,5],[2,7],[7,10]]
            att3_clique=get_clique(15,3,100)  #[[0,1,2],[6,7,8],[9,10,11],[12,13,14],[3,5,7],[9,11,13],[2,13,15],[3,5,9],[4,5,15],[8,11,13],[9,10,15]]
            att4_clique=get_clique(15,4,100)     #[[0,1,2,3],[4,5,6,7],[8,9,10,11],[2,4,6,8],[8,10,12,14],[3,6,8,11],[4,6,7,9],[2,4,7,8],[5,8,9,11],[3,5,6,14]]
            att6_clique=get_clique(15,6,100)   #[[0,1,2,3,4,5],[6,7,8,9,10,11]]
            att8_clique=get_clique(15,8,100)    #[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]
            att7_clique=get_clique(15,7,100)
            att5_clique=get_clique(15,5,10)
            att10_clique=get_clique(15,10,10)
            att12_clique=get_clique(15,12,10)
            #print(att2_clique)
            #print(att3_clique)
            #print(att4_clique)
            #print(att6_clique)
            #print(att8_clique)
            
            if file_id==4:
            
                mean_err=0
                i=0
                for eachclique in att2_clique:
                    curr_time=time.time()
                    some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
                    #some_list,pro=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist_sparse, bitsum_list3, f, dt)
                    elapse=time.time()-curr_time
                     
                    true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
                    #print('true:',true_pro)
                    #print('esti2:',pro)
                    i+=1
            
                    err=l2_err(pro, true_pro)
                    mean_err+=err
                    #print(i,err,elapse)
                mean_err=1.0*mean_err/len(att2_clique)
                print('2-way',mean_err)
                     
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
                
                
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
