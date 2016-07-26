import Get_Rappor
import Get_Params
import numpy
import copy
from copy import copy, deepcopy
from itertools import product,combinations
from random import randint


def rappor_prob(x_rappor,x_signal,f=0.0):
    x_rappor=numpy.array(x_rappor)
    x_signal=numpy.array(x_signal)
 
    prob=numpy.prod(numpy.power((0.5*f),x_signal*x_rappor+(1-x_signal)*(1-x_rappor))*numpy.power((1.0-0.5*f),(1-x_signal)*x_rappor+x_signal*(1-x_rappor)),1)

#     for i in range(len(x_signal)) :
#         B=x_signal[i]
#         b=x_rappor[i]
#         z=((((0.5*f)**B)*((1.0-0.5*f)**(1-B)))**b)*((((0.5*f)**(1-B))*((1.0-0.5*f)**B))**(1-b))
#         #print(z)
#         prob=prob*z
    return prob
# x=numpy.array([[0,1,0],[0,1,0]])
# y=numpy.array([[0,1,1],[0,1,0]])
# print(rappor_prob(x, y, 0.1))

def joint_prob(i,j,p,f,x_rappor,x_signal,y_rappor,y_signal):
    prob=p[i][j]*rappor_prob(x_rappor, x_signal, f)*rappor_prob(y_rappor, y_signal, f)
    return prob

def get_bayes(x_rappor,y_rappor,x_signal_list,y_signal_list,p,f,i,j):
    
    fenzi=joint_prob(i,j,p,f,x_rappor,x_signal_list[i],y_rappor,y_signal_list[j])
    fenmu=0.0
    
    pp=numpy.array([[p for i in range(len(x_signal_list))]for j in range(len(y_signal_list))])
    fenmu=numpy.dot(rappor_prob(x_rappor, x_signal_list, f)*rappor_prob(y_rappor, y_signal_list, f),)
    
    for i in range(len(x_signal_list)):
        for j in range(len(y_signal_list)):
            fenmu=fenmu+joint_prob(i, j, p, f, x_rappor, x_signal_list[i], y_rappor, y_signal_list[j])      
    bay_pro=fenzi/fenmu
    return bay_pro

    
    
def estimate_2d(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,f,dt=0.001):
    m=len(att1signal_list)
    n=len(att2signal_list)
    N=len(att1data_rappor_list)
    p=[[(1.0/(m*n)) for i in range(n)]for j in range(m)]
    #print('p:',p)
    temp=[[0.0 for i in range(n)]for j in range(m)]
    deltap=deepcopy(p)
    
    for i in range(10):
        temp=deepcopy(p)
        #print('temp',temp)
        
        maxp=numpy.max(deltap)
        #print('maxp',maxp)
        if maxp<(dt/(m*n)):
            break
        pass        
        deltap=deepcopy(p)
        for i in range(m):
            for j in range(n):
                print('Estimate',i,'of',m,j,'of',n)
                p=get_baye(att1data_rappor_list, att2data_rappor_list, att1signal_list, att2signal_list, p, f, N)
                p[m-i-1][n-j-1]=0
                deltap[i][j]=deepcopy(abs(p[i][j]-temp[i][j]))
        
        #print('p',p)
        #print('deltap',deltap)    
        print('p:',p)
    return list(p)

def get_baye_fenzi(x_rappor_list, y_rappor_list, x_signal_list, y_signal_list,f):
    a1=numpy.tile(x_signal_list,(len(x_rappor_list),1))
    a2=numpy.tile(y_signal_list,(len(y_rappor_list),1))
    att1=numpy.tile(x_rappor_list,(len(x_signal_list),1))
    att2=numpy.tile(y_rappor_list,(len(y_signal_list),1))
    fenzi=rappor_prob(att1, a1, f)*rappor_prob(att2, a2, f)
    return fenzi

def get_baye(x_rappor_list, y_rappor_list, x_signal_list, y_signal_list,p,f,N):
    m=len(x_signal_list)
    n=len(y_signal_list)
    he=deepcopy(p)
#     for i in range(m):
#         for j in range(n):
            #print('Estimate',i,'of',m,j,'of',n)
    he=(numpy.sum(p*get_baye_fenzi(x_rappor_list, y_rappor_list, x_signal_list, y_signal_list, i, j, f)))
    fenzi=he/numpy.sum(he)
N=6
i=1
j=2
f=0.1
att1data_rappor_list=numpy.array([[0,1],[1,0],[1,1],[0,0],[1,1],[0,1]])
att2data_rappor_list=numpy.array([[0,1,1],[1,0,1],[1,1,0],[0,0,0],[1,1,1],[0,1,1]])
att1signal_list=numpy.array([[0,0],[0,1],[1,0],[1,1]])
att2signal_list=numpy.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

b1=att1data_rappor_list.tolist()
b2=att2data_rappor_list.tolist()

import itertools
z=[]
for i in itertools.product(b1,b2):
    z.append(list(i))
print(z)
  
a1=numpy.tile(att1signal_list,(N,1))
a2=numpy.tile(att2signal_list,(N,1))
attdata1=numpy.tile(att1data_rappor_list,(4,1)) 
attdata2=numpy.tile(att2data_rappor_list,(8,1)) 
p1=rappor_prob(attdata1, a1, f)
p2=rappor_prob(attdata2, a2, f)
z=get_baye_fenzi(att1data_rappor_list, att2data_rappor_list, att1signal_list, att2signal_list, f)
print(z)
  
a1=numpy.tile(att1signal_list,(N,1))
a2=numpy.tile(att2signal_list,(N,1))
p1=rappor_prob(att1data_rappor_list, a1, f)
p2=rappor_prob(att2data_rappor_list, a2, f)
z=get_baye_fenzi(att1data_rappor_list, att2data_rappor_list, a1, a2, f)
print(z)
#  
# a1=numpy.tile(att1signal_list[0],(N,1))
# a2=numpy.tile(att2signal_list[2],(N,1))
# p1=rappor_prob(att1data_rappor_list, a1, f)
# p2=rappor_prob(att2data_rappor_list, a2, f)
# z=get_baye_fenzi(att1data_rappor_list, att2data_rappor_list, a1, a2, f)
# print(z)

























###################################################################################################################################################################################################33#
def list_paste(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list):
    att3data_rappor_list=[x+y for x,y in zip(att1data_rappor_list,att2data_rappor_list)]
    att3signal_list=[]
    m=len(att1signal_list)
    n=len(att2signal_list)
    
    for i in range(m):
        for j in range(n):
            att3signal_list.append(att1signal_list[i]+att2signal_list[j])
    
    return att3data_rappor_list,att3signal_list

def rappor_list_paste(*attdata_rappor_list):   ####################################  for arbitrary lists
    m=len(attdata_rappor_list[0])
    temp_rappor_list=[[]for i in range(m)]
    for each_attdata_rappor_list in attdata_rappor_list:
        attcombine_rappor_list=[x+y for x,y in zip(temp_rappor_list,each_attdata_rappor_list)]
        temp_rappor_list=attcombine_rappor_list
    return attcombine_rappor_list

def list_product(att1signal_list,att2signal_list):
    product_list=[]
    for att12_product in product(att1signal_list,att2signal_list):
        temp=[]
        for item in att12_product:
            temp.extend(item)
        product_list.append(temp)
    return product_list

def signal_list_paste(*attsignal_list):      ###################################### for arbitrary lists
    n=len(attsignal_list)
    signal_combine_list=[]
    temp_signal_list=attsignal_list[0]
    for i in range(n-1):
        signal_combine_list=list_product(temp_signal_list, attsignal_list[i+1])
        temp_signal_list=signal_combine_list
    return signal_combine_list

def att_combin(bit_list,bit_cand_list,row_list,loc_list):
    #loc_list specify the index of the attributes needed to be combined
    rappor_para_string=()
    signal_para_string=()
    row_string=()
    for loc in loc_list:
        rappor_para_string=rappor_para_string+(bit_list[loc],)
        signal_para_string=signal_para_string+(bit_cand_list[loc],)
        row_string=row_string+(row_list[loc],)
        
    att_rappor_list_combine=rappor_list_paste(*rappor_para_string)
    att_signal_list_combine=signal_list_paste(*signal_para_string)
    att_row_list_combine=signal_list_paste(*row_string)
    return att_rappor_list_combine,att_signal_list_combine,att_row_list_combine





############################################################################# Example for list process, i.e., listpaste, listproduct  

# list1=[[1,2],[3,4]]
# list2=[[5,6],[7,8]]
# list3=[[9,10],[11,12]]
# z=[list1,list2,list3]
# print(z) 
# print(rappor_list_paste(list1,list2,list3))
# print(list_paste(list1, list2,list2,list3))
# print(list_product(list1,list2))
# list_t=list_product(list1,list2)
# list_s=list_product(list_t,list3)
# print(list_s)
# print(signal_list_paste(list1,list2,list3))
#  
# att1,att2,att3=att_combin(z,z,z, [0,1,2])
# print(att1)
# print(att2)
# print(att3)
#print(rappor_list_paste())
        
    
######################################################################################    EXAMPLE for attributes combination  ###################################################################################################  
bit_cand_list,bit_list,bitsum_list=Get_Rappor.rappor_process(4, 8, 2, 0.01)
# att_num,node_num,rowlist,multilist=Get_Params.get_file_info(4)
# loc_list=[0,2,3,5,8]
# att_rappor_list_combine,att_signal_list_combine,att_row_list_combine=att_combin(bit_list, bit_cand_list, rowlist, loc_list)
# print(att_rappor_list_combine)
# print(att_signal_list_combine)
# print(att_row_list_combine)
######################################################################################################################################################################################################
# bit_list=map(list, zip(*bit_list))
# 
p02=estimate_2d(bit_list[0], bit_list[3], bit_cand_list[0], bit_cand_list[3], 0.01, 0.01)
# att12data_rappor_list,att12signal_list=list_paste(bit_list[0], bit_list[2], bit_cand_list[0], bit_cand_list[2])
#   
# p02_3=estimate_2d(att12data_rappor_list, bit_list[3], att12signal_list, bit_cand_list[3], 0.01, 0.0001)
#print(p02)
# print(p02_3)
#p04=estimate_2d(bit_list[0], bit_list[0], bit_cand_list[0], bit_cand_list[0], 0.01, 0.0001)
#print(p04)

#lasso_cf=Get_Rappor.lasso_regression(bit_cand_list,bitsum_list)
#print(lasso_cf)






