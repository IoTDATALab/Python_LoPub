import rappor2
import rappor
import copy
import random
import csv
import os
#import pandas
#import numpy

from random import sample
from Tkconstants import CHAR
from rappor import get_bloom_bits
from collections import Counter
#from re import MULTILINE



def get_file_info(input_id, readlimit,samplerate):
    ##Params:
    ##input_id: Specify the input file number
    ##read_limit: Specify the number of rows reading
    
    input_id=str(input_id)
    input_data='C:\Users\Ren\workspace2\DisHD\data\Data'+input_id+'-coarse.dat'
    input_domain='C:\Users\Ren\workspace2\DisHD\data\Data'+input_id+'-coarse.domain'
##############################################################################################
#Get the attributes number and nodes number
    fd=open(input_domain,'r')
    head_line=fd.readline()
    readrow=head_line.split(" ")
    #print(readrow)
    att_num=int(readrow[0])
    node_num=int(readrow[1])
    if (node_num>=readlimit):
        node_num=readlimit
    ##print(att_num,node_num)
    
    ##########################################################################################
    #Get the attributes info in the domain file
    rowlist = [[] for row in range(att_num)]
    newlist=[[] for row in range(att_num)]
    
    i=0
    while 1:
        line = fd.readline()
        if not line:
            break
        pass # do something
        
        if i>=readlimit:
            break
        pass
    
        line=line.strip("\n")
        readrow=line.split(" ")
        #print(readrow)
        start_x=0
        each_code=0  #transfer non-number data into numbers
        for eachit in readrow:
            start_x=start_x+1
            eachit.rstrip()
            if start_x>3:
                neweach=eachit
                
                #print(neweach)
                newlist[i].append(neweach)
                #rowlist[i].append(eachit)
                rowlist[i].append(str(each_code))#transfer non-number data into numbers
                each_code=each_code+1#transfer non-number data into numbers
            #print(eachit)
        i=i+1
    fd.close()
    #print(rowlist)
    #print(newlist)
    #newlist.remove([])
    #rowlist.remove([])
    
    ############################################################################
    #Get the data in the .dat file
    multilist = [[] for row in range(node_num)]
    fp=open(input_data,"r")
    fp.readline();  ###just for skip the header line
    i=0
    entry={}
    while 1:
        line = fp.readline()
        if not line:
            break
        pass # do something
        
        if i>=readlimit:
            break
        pass
    
        line=line.strip("\n")
        temp=line.split(",")
        #print(node_num,i, 'after split:', temp)
        entry[i]=temp
        iii=0#transfer non-number data into numbers
        for eachit in temp:
            each_index=newlist[iii].index(eachit)#transfer non-number data into numbers
            iii=iii+1#transfer non-number data into numbers
            multilist[i].append(str(each_index))#transfer non-number data into numbers
            #multilist[i].append(eachit)
        i=i+1
    fp.close()
    #print(multilist)
    
    samplesize=int(node_num*samplerate)
    multilist=random.sample(multilist,samplesize)
    true_node_num=node_num
    node_num=samplesize
    
#     node_limit=3
#     rowlist=rowlist[0:node_limit]
#     multilist=map(list,zip(*multilist))
#     multilist=multilist[0:node_limit]
#     multilist=map(list,zip(*multilist))
#     att_num=node_limit
    
    
    return att_num,node_num,true_node_num,rowlist,multilist



###############################################################################################################################################
def get_static_info(att_num,node_num,rowlist,multilist):
    newlist= [[]for i in range(att_num)]
    freqrow1=[[]for i in range(att_num)]
    freqnum1=[[]for i in range(att_num)]
    freqrate1=[[] for i in range(att_num)]
    freqrow2=[[] for row in range(att_num*(att_num-1)/2)] 
    freqnum2=[[] for row in range(att_num*(att_num-1)/2)] 
    freqrate2=[[] for row in range(att_num*(att_num-1)/2)] 
    
    ii=0
    
    for coli in range(att_num):
        for rowi in range(node_num):
            newlist[coli].append(multilist[rowi][coli]) # may be useful for sum!
        freqrow1[coli]=Counter(newlist[coli])
        for eachitem in rowlist[coli]:
            freqnum1[coli].append(freqrow1[coli][eachitem])
        for i1 in range(len(freqnum1[coli])):
            freqrate1[coli].append(1.0*freqnum1[coli][i1]/node_num)       
    
        for colj in range(coli+1, att_num):
            possible_combine=[]
            for eachi in rowlist[coli]:
                for eachj in rowlist[colj]:
                    possible_combine.append(str(eachi)+'|'+str(eachj))
            rowwriterr=[]
            for rowi in range(node_num):
                strtemp=str(multilist[rowi][coli])+'|'+str(multilist[rowi][colj])
                rowwriterr.append(strtemp)
            freqrow2[ii]=Counter(rowwriterr)
            for eachitem in possible_combine:
                freqnum2[ii].append(freqrow2[ii][eachitem])
            for i2 in range(len(freqnum2[ii])):
                freqrate2[ii].append(1.0*freqnum2[ii][i2]/node_num)
            ii=ii+1
               
    return freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist



###############################################################################################################################################
def set_rappor_params(num_bloombits,num_hash,col,f):
    params=rappor.Params()
    params.num_hashes = num_hash         # Number of bloom filter hashes
    params.num_bloombits = num_bloombits
    params.prob_f = f
    params.prob_p = 0.0
    params.prob_q = 1.0
    rand=rappor.MockRandom([0.0, 0.6, 0.0], params)
    secret=str(col)
    #print(rand)
    #secret=random
    e=rappor2.Encoder(params, 0, secret, rand)
    #print(e.params.prob_f)
    return e
    
def get_S(str_x,e):
    bit_array=[]
    bit_str=e._internal_encode(str_x)[0]    
    for each_char in bit_str:
        bit_array.append(int(each_char))
    #print(bit_array)
    return bit_array

def get_B(str_x,e):
    bit_array=[]
    bit_str=e._internal_encode(str_x)[1]    
    for each_char in bit_str:
        bit_array.append(int(each_char))  
    return bit_array

# for i in range(10):
#     set_rappor_params(32, 4, 0.5)

# att_num,node_num,rowlist,multilist=get_file_info(4,10000)
# freqrow1,freqnum1,freqrate1,freqrow2,freqnum2,freqrate2,newlist=get_static_info(att_num, node_num, rowlist, multilist)
#  
# print(freqrate1)
# print(freqrate2)
