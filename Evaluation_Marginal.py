import Get_Params
import Get_Rappor
import numpy
import itertools
import random
import time

from JunctionTree import independent_marginal2,independent_marginal
from Estimate_Joint_Distribution import true_joint_distribution, unfold_pro_list
from numpy import power

file_id=2
fai_C=0.2    #from 0.2, 0.3, 0.4, 0.5
f=0.5   # from 0.1, 0.2, 0.3, 0.4, 0.5  *********
bloombit=32
hashbit=2
dt=0.01
readlimit=50000
samplerate=0.0215  # from 0.01, 0.05, 0.1, 0.5, 1 0.0215

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
        

att_num1,node_num1,true_node_num1,rowlist1,multilist1=Get_Params.get_file_info(file_id,readlimit,1.0)
att_num2,node_num2,true_node_num2,rowlist2,multilist2=Get_Params.get_file_info(file_id,readlimit,samplerate)

bit_cand_list2,bit_list2,bitsum_list2=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num2,node_num2,true_node_num2,rowlist2,multilist2)

sparse_rate=0.15
rowlist_sparse,multilist_sparse=Get_Rappor.Get_rid_sparse(bit_cand_list2,bitsum_list2,att_num2,node_num2,true_node_num2,rowlist2,multilist2,sparse_rate)
#print(rowlist_sparse)
bit_cand_list3,bit_list3,bitsum_list3=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num2,node_num2,true_node_num2,rowlist_sparse,multilist_sparse)

freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=Get_Params.get_static_info(att_num2, node_num2, rowlist_sparse, multilist_sparse)
print('finish basis!')
#print(multilist_sparse)

att_clique=range(att_num2)
ini_list2=list(itertools.combinations([1,2,3,4],2))
z=[list(eachtuple) for eachtuple in ini_list2]


att2_clique=get_clique(15, 2, 10)   #[[0,1],[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[2,5],[2,7],[7,10]]
att3_clique=get_clique(15,3,10)  #[[0,1,2],[6,7,8],[9,10,11],[12,13,14],[3,5,7],[9,11,13],[2,13,15],[3,5,9],[4,5,15],[8,11,13],[9,10,15]]
att4_clique=get_clique(15,4,10)     #[[0,1,2,3],[4,5,6,7],[8,9,10,11],[2,4,6,8],[8,10,12,14],[3,6,8,11],[4,6,7,9],[2,4,7,8],[5,8,9,11],[3,5,6,14]]
att6_clique=get_clique(15,6,10)   #[[0,1,2,3,4,5],[6,7,8,9,10,11]]
att8_clique=get_clique(15,8,10)    #[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]
att7_clique=get_clique(15,7,10)
att5_clique=get_clique(15,5,10)
att10_clique=get_clique(15,10,10)
att12_clique=get_clique(15,12,10)
print(att2_clique)
print(att3_clique)
print(att4_clique)
print(att6_clique)
print(att8_clique)

if file_id==4:
    mean_err12=0
    mean_err22=0
    mean_err32=0
    i=0
    for eachclique in att2_clique:
        #print(eachclique)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
         
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,200.0/true_node_num1,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        print('esti1:',pro1)
        print('esti2:',pro2)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err12+=err1
        mean_err22+=err2
        mean_err32+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    mean_err12=1.0*mean_err12/len(att2_clique)
    mean_err22=1.0*mean_err22/len(att2_clique)
    mean_err32=1.0*mean_err32/len(att2_clique)
    print('2-way',mean_err12,mean_err22,mean_err32)
         
    mean_err13=0
    mean_err23=0
    mean_err33=0
    i=0
    for eachclique in att3_clique:
        #print(eachclique)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num1,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        print('esti1:',pro1)
        print('esti2:',pro2)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err13+=err1
        mean_err23+=err2
        mean_err33+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    mean_err13=1.0*mean_err13/len(att3_clique)
    mean_err23=1.0*mean_err23/len(att3_clique)
    mean_err33=1.0*mean_err33/len(att3_clique)
    print('3-way',mean_err13,mean_err23,mean_err33)
     
    mean_err14=0
    mean_err24=0
    mean_err34=0
    i=0
    for eachclique in att4_clique:
        #print(eachclique)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
         
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        print('esti1:',pro1)
        print('esti2:',pro2)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err14+=err1
        mean_err24+=err2
        mean_err34+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    mean_err14=1.0*mean_err14/len(att4_clique)
    mean_err24=1.0*mean_err24/len(att4_clique)
    mean_err34=1.0*mean_err34/len(att4_clique)
    print('4-way',mean_err14,mean_err24,mean_err34)
     
     
    mean_err15=0
    mean_err25=0
    mean_err35=0
    i=0
    for eachclique in att5_clique:
        #print(eachclique)
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
         
         
        print('esti1:',pro1)
        print('esti2:',pro2)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err15+=err1
        mean_err25+=err2
        mean_err35+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    mean_err15=1.0*mean_err15/len(att5_clique)
    mean_err25=1.0*mean_err25/len(att5_clique)
    mean_err35=1.0*mean_err35/len(att5_clique)
    print('5-way',mean_err15,mean_err25,mean_err35)
    
    
    mean_err16=0
    mean_err26=0
    mean_err36=0
    i=0
    for eachclique in att6_clique:
        #print(eachclique)
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
        
        
        print('esti1:',pro1)
        print('esti2:',pro2)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err16+=err1
        mean_err26+=err2
        mean_err36+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    #mean_err1=1.0*mean_err1/len(att2_clique)
    mean_err26=1.0*mean_err26/len(att6_clique)
    mean_err36=1.0*mean_err36/len(att6_clique)
    print('6-way',mean_err16,mean_err26,mean_err36)
    
    
    mean_err17=0
    mean_err27=0
    mean_err37=0
    i=0
    for eachclique in att7_clique:
        #print(eachclique)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
        
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        #print('true:',true_pro)
        #print('esti1:',pro1)
        #print('esti2:',pro2)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err17+=err1
        mean_err27+=err2
        mean_err37+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    mean_err17=1.0*mean_err17/len(att7_clique)
    mean_err27=1.0*mean_err27/len(att7_clique)
    mean_err37=1.0*mean_err37/len(att7_clique)
    print('7-way',mean_err17,mean_err27,mean_err37)
    
    
    mean_err18=0
    mean_err28=0
    mean_err38=0
    i=0
    for eachclique in att8_clique:
        #print(eachclique)
        curr_time2=time.time()
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist2, bitsum_list3, f, dt)
        elapse2=time.time()-curr_time2
        curr_time1=time.time()
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        elapse1=time.time()-curr_time1
        
        true_list,true_pro=true_joint_distribution(multilist1, rowlist1, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        #print('true:',true_pro)
        #print('esti1:',pro1)
        #print('esti2:',pro2)
        #print('esti3:',pro3)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=l2_err(pro2, true_pro)
        err3=l2_err(pro3,true_pro)
        mean_err18+=err1
        mean_err28+=err2
        mean_err38+=err3
        print(i,err1,err2,err3,elapse1,elapse2)
    mean_err18=1.0*mean_err18/len(att2_clique)
    mean_err28=1.0*mean_err28/len(att8_clique)
    mean_err38=1.0*mean_err38/len(att8_clique)
    print('8-way',mean_err18,mean_err28,mean_err38)
    
else:
    sum_err12=0
    sum_err22=0
    sum_err32=0
    i=0
    for eachclique in att2_clique:
        #print(eachclique)
        true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,200.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist_sparse, bitsum_list3, f, dt)
        print('esti2:',pro2)
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        #pro1=pro2
        
        
        print('esti1:',pro1)
        
        #print('esti3:',pro3)
        i+=1
        err1=l2_err(pro1,true_pro)
        err2=get_avd(pro2, true_pro)
        err3=get_avd(pro3,true_pro)
        sum_err12+=err1
        sum_err22+=err2
        sum_err32+=err3
        print(i,err1,err2,err3)
    mean_err12=sum_err12/len(att2_clique)
    mean_err22=sum_err22/len(att2_clique)
    mean_err32=sum_err32/len(att2_clique)
    print('2-way',mean_err12,mean_err22,mean_err32)
    
    sum_err13=0
    sum_err23=0
    sum_err33=0
    i=0
    for eachclique in att3_clique:
        #print(eachclique)
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist_sparse, bitsum_list3, f, dt)
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        
        true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        print('esti1:',pro1)
        print('esti2:',pro2)
        #print('esti3:',pro3)
        i+=1
        err1=get_avd(pro1,true_pro)
        err2=get_avd(pro2, true_pro)
        err3=get_avd(pro3,true_pro)
        sum_err13+=err1
        sum_err23+=err2
        sum_err33+=err3
        print(i,err2,err3)
    mean_err13=sum_err13/len(att2_clique)
    mean_err23=sum_err23/len(att3_clique)
    mean_err33=sum_err33/len(att3_clique)
    print('3-way',mean_err13,mean_err23,mean_err33)
    
    
    sum_err14=0
    sum_err24=0
    sum_err34=0
    i=0
    for eachclique in att4_clique:
        #print(eachclique)
        some_list,pro2=independent_marginal2(eachclique, bit_list3, bit_cand_list3, rowlist_sparse, bitsum_list3, f, dt)
        some_list,pro1=independent_marginal(eachclique, bit_list3, bit_cand_list3, rowlist2, f, dt)
        
        true_list,true_pro=true_joint_distribution(multilist_sparse, rowlist_sparse, eachclique)
        pro3temp=numpy.array(true_pro)+numpy.random.laplace(0,400.0/true_node_num2,len(true_pro))
        pro3=pro3temp.tolist()
        print('true:',true_pro)
        print('esti1:',pro1)
        print('esti2:',pro2)
        #print('esti3:',pro3)
        i+=1
        err1=get_avd(pro1,true_pro)
        err2=get_avd(pro2, true_pro)
        err3=get_avd(pro3,true_pro)
        sum_err14+=err1
        sum_err24+=err2
        sum_err34+=err3
        print(i,err1,err2,err3)
    mean_err14=sum_err14/len(att2_clique)
    mean_err24=sum_err24/len(att3_clique)
    mean_err34=sum_err34/len(att3_clique)
    print('4-way',mean_err14,mean_err24,mean_err34)
#     mean_err=0
#     i=0
#     for eachclique in att2_clique:
#         some_list,pro=independent_marginal(eachclique, bit_list, bit_cand_list, rowlist,bitsum_list, f, dt)
#         true_list,true_pro=true_joint_distribution(multilist, rowlist, eachclique)
#         print('true:',true_pro)
#         i+=1
#         err=get_avd(pro, true_pro)
#         mean_err+=err
#         print(i,err)
#     mean_err=1.0*mean_err/len(att2_clique)
#     print(mean_err)
#     
#     
#     mean_err=0
#     i=0
#     for eachclique in att3_clique:
#         some_list,pro=independent_marginal(eachclique, bit_list, bit_cand_list, rowlist,bitsum_list, f, dt)
#         true_list,true_pro=true_joint_distribution(multilist, rowlist, eachclique)
#         print('true:',true_pro)
#         i+=1
#         err=get_avd(pro, true_pro)
#         mean_err+=err
#         print(i,err)
#     mean_err=1.0*mean_err/len(att3_clique)
#     print(mean_err)
#         
        
    
    
