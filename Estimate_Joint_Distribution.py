import Get_Rappor
import Get_Params
import numpy
import copy
from copy import copy, deepcopy
from itertools import product,combinations
from random import randint
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.linear_model.coordinate_descent import enet_path
from sklearn.linear_model import Lasso, LinearRegression



def rappor_prob(x_rappor=[],x_signal=[],f=0.0):
    x_rappor=numpy.array(x_rappor)
    x_signal=numpy.array(x_signal)
    #print(x_rappor)
    #print(x_signal)
    prob=1.0
    prob=numpy.prod(numpy.power(1-(0.5*f),x_signal*x_rappor+(1-x_signal)*(1-x_rappor))*numpy.power((0.5*f),(1-x_signal)*x_rappor+x_signal*(1-x_rappor)))
    
#     for i in range(len(x_signal)) :
#         B=x_signal[i]
#         b=x_rappor[i]
#         z=((((0.5*f)**B)*((1.0-0.5*f)**(1-B)))**b)*((((0.5*f)**(1-B))*((1.0-0.5*f)**B))**(1-b))
#         #print(z)
#         prob=prob*z
    return prob


def joint_prob(i,j,p,f,x_rappor,x_signal,y_rappor,y_signal):
    prob=p[i][j]*rappor_prob(x_rappor, x_signal, f)*rappor_prob(y_rappor, y_signal, f)
    return prob

#get_bayes(att1data_rappor_list[kk], att2data_rappor_list[kk], att1signal_list, att2signal_list, p, f, i, j)

def get_bayes(x_rappor,y_rappor,x_signal_list,y_signal_list,p,f,i,j):
    
    fenzi=joint_prob(i,j,p,f,x_rappor,x_signal_list[i],y_rappor,y_signal_list[j])
    fenmu=0.0
    for i in range(len(x_signal_list)):
        for j in range(len(y_signal_list)):
            fenmu=fenmu+joint_prob(i, j, p, f, x_rappor, x_signal_list[i], y_rappor, y_signal_list[j])      
    #print(fenzi,fenmu)
    bay_pro=fenzi/fenmu
    return bay_pro

def ad_lasso(att_cand,sum_cand_list):
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Lasso
    
    em_lasso_cf=[] 
    arr_att_cand=numpy.array(att_cand[0])
    arr_sum_cand_list=numpy.matrix(sum_cand_list[0])
#     print(numpy.shape(arr_att_cand),numpy.shape(arr_sum_cand_list))
#     print(arr_att_cand)
#     print(arr_sum_cand_list)
    Y=arr_sum_cand_list.T
    X=arr_att_cand.T
    
        
#     X, y, coef = make_regression(n_samples=306, n_features=8000, n_informative=50,
#                         noise=0.1, shuffle=True, coef=True, random_state=42)
#     
#     X /= np.sum(X ** 2, axis=0)  # scale features
    
    alpha = 0.1
    
#     g = lambda w: np.sqrt(np.abs(w))
#     gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)
    
    #Or another option:
    ll = 0.01
    g = lambda w: np.log(ll + np.abs(w))
    gprime = lambda w: 1. / (ll + np.abs(w))
#     
    n_samples, n_features = X.shape
    p_obj = lambda w: 1. / (2 * n_samples) * np.sum((Y - np.dot(X, w)) ** 2) \
                      + alpha * np.sum(g(w))
    
    weights = np.ones(n_features)
    n_lasso_iterations = 5
    
    for k in range(n_lasso_iterations):
        X_w = X / weights[np.newaxis, :]
        #clf=LinearRegression()
        clf = Lasso(alpha=alpha, fit_intercept=False)
        clf.fit(X_w, Y)
        coef_ = clf.coef_ / weights
        weights = gprime(coef_)
        #print p_obj(coef_)  # should go down
        #print(coef_)
        coef=coef_
        ratio=coef/(sum(coef))
        print(k,'coef',ratio.tolist())
    em_lasso_cf.append(ratio.tolist())
    print('coef',em_lasso_cf)
    return em_lasso_cf
    #print np.mean((clf.coef_ != 0.0) == (coef != 0.0))


def em_lasso(att_cand,sum_cand_list):
    em_lasso_cf=[] 
    arr_att_cand=numpy.array(att_cand[0])
    arr_sum_cand_list=numpy.matrix(sum_cand_list[0])
#     print(numpy.shape(arr_att_cand),numpy.shape(arr_sum_cand_list))
#     print(arr_att_cand)
#     print(arr_sum_cand_list)
    Y=arr_sum_cand_list.T
    X=arr_att_cand.T

    lasso_test=lasso.lasso(Y,X)
    lasso_test.EM(5000)
    print('sigma',lasso_test.sigma2)

    #set_printoptions(precision=3)

    em_beta = lasso_test.beta.T
    em_beta[abs(em_beta / em_beta.max()) < 0.0005] = 0.0
    OLS = numpy.linalg.solve(numpy.dot(X.T, X), numpy.dot(X.T, Y))

    from sklearn import linear_model
    #import sys
    #YY=numpy.ndarray(Y)

    clf = linear_model.Lasso(alpha=.1)
   # print(numpy.shape(X),)
    clf.fit(X, Y.A1)
    coef=clf.coef_
    ratio=coef/(sum(coef))
    em_lasso_cf.append(ratio.tolist())
    print(em_lasso_cf)
    print('coef',clf.coef_)
    print('ols',OLS.T)
    print('beta',em_beta)
         
    return em_lasso_cf


def estimate_2d2(att1signal_list,att2signal_list,bitsum_list,clique):
    att_cand=[list_product(att1signal_list, att2signal_list)]
    sum_cand=[]
    for eachitem in clique:
        sum_cand.extend(bitsum_list[eachitem])
    sum_cand_list=[sum_cand]
    #zz=ad_lasso(att_cand, sum_cand_list)
    proe=Get_Rappor.lasso_regression(att_cand, sum_cand_list)
    protemp=proe[0]
    #protemp=zz[0]
    lenpro=len(protemp)
    leng1=len(att1signal_list)
    leng2=len(att2signal_list)
    ptemp=[[]for i in range(leng1)]
    for i in range(leng1):
        for j in range(leng2):
            ptemp[i].append(protemp[(i)*leng2+j])
    return ptemp

 
 
 
def estimate_2d1(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,bitsum_list,clique,f,dt=0.001):
#def estimate_2d(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,f,dt=0.001):
#     print(att1signal_list)
#     print(att2signal_list)
#     print(att1data_rappor_list[1])
#     print(att2data_rappor_list[1])
    
    
    m=len(att1signal_list)
    n=len(att2signal_list)
    N=len(att1data_rappor_list)
    #print(m,n,N)
    p_cond=numpy.array([[0.0 for i in range(N)] for j in range(m*n)])
    #print(p_cond,p_cond.sum())
    p=estimate_2d2(att1signal_list, att2signal_list, bitsum_list, clique)
    #p=[[(1.0/(m*n)) for j in range(n)]for i in range(m)]  #initial value
    p_result=numpy.array([[0.0 for j in range(n)]for i in range(m)])
    maxp=1.0 
    tt=0

    for kk in range(N):  
        for i in range(m):
                for j in range(n):
                    p_cond[i*(n)+j][kk]=rappor_prob(att1data_rappor_list[kk], att1signal_list[i], f)*rappor_prob(att2data_rappor_list[kk], att2signal_list[j], f)                      
                #print (p_cond[i*(n)+j][kk])
    #for kk in range(N):
    #print('sum',p_cond.sum())
    
    while(abs(p_result.max()-maxp)>0.001):
    #while(tt<=10):
        print(tt)
        tt+=1
        maxp=p_result.max()
        p_poster_temp=[[0.0 for i in range(n)] for j in range(m)]
        for kk in range(N):
             
            fenmu=0.0
            fenzi=0.0
            
            for i2 in range(m):
                for j2 in range(n):
                    fenmu+=p[i2][j2]*p_cond[i2*(n)+j2][kk]
                             
                             
            for i in range(m):
                for j in range(n):
                    fenzi=p[i][j]*p_cond[i*(n)+j][kk]
                    p_poster_temp[i][j]+=fenzi/fenmu
                     
                     
        for i in range(m):
            for j in range(n):
                p[i][j]=p_poster_temp[i][j]/N
  
#         sump=0.0
#         for i in range(m):
#             for j in range(n):
#                 sump+=p[i][j]
#         for i in range(m):
#             for j in range(n):
#                 p_result[i][j]=p[m-1-i][n-1-j]/sump    
           
        #if maxp<p_result.max():
        #print('p_EM:',p_result)
    return list(p_result) 
 
 
   
def estimate_2dnew(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,bitsum_list,clique,f,dt=0.001):
    m=len(att1signal_list)
    n=len(att2signal_list)
    N=len(att1data_rappor_list)
    p_cond=numpy.array([[0.0 for j in range(N)] for i in range(m*n)])
    #p_cond=numpy.array([[0.0 for j in range(m*n)] for i in range(N)])
    p=[[(1.0/(m*n)) for j in range(n)]for i in range(m)]
    #p_result=[[0.0 for j in range(n)]for i in range(m)]
    p_result=numpy.array([[0.0 for j in range(n)]for i in range(m)])

    maxp=1.0 
    tt=0

    for kk in range(N):  
        for i in range(m):
            for j in range(n):
                    #print(att1signal_list)
                    #print(att2signal_list)
                p_cond[i*(n)+j][kk]=rappor_prob(att1data_rappor_list[kk], att1signal_list[i], f)*rappor_prob(att2data_rappor_list[kk], att2signal_list[j], f)                      
                #print (p_cond[i*(n)+j][kk])
    #for kk in range(N):
    #print('sum',p_cond.sum())
    #print('conditon',p_cond)
    #while(abs(p_result.max()-maxp)>0.001):
    while(tt<=15):
        #print(tt)
        tt+=1
        maxp=p_result.max()
        p_poster_temp=[[0.0 for j in range(n)] for i in range(m)]
        for kk in range(N):
             
            fenmu=0.0
            fenzi=0.0
            
            for i2 in range(m):
                for j2 in range(n):
                    fenmu+=p[i2][j2]*p_cond[i2*(n)+j2][kk]
                             
                             
            for i in range(m):
                for j in range(n):
                    fenzi=p[i][j]*p_cond[i*(n)+j][kk]
                    p_poster_temp[i][j]=p_poster_temp[i][j]+fenzi/fenmu
        
                     
                     
        for i in range(m):
            for j in range(n):
                p[i][j]=p_poster_temp[i][j]/N
        
        #print('p',p)
        sump=0.0
        for i in range(m):
            for j in range(n):
                sump=sump+p[i][j]
        for i in range(m):
            for j in range(n):
                p_result[i][j]=p[m-1-i][n-1-j]/sump    
        #print('p_restult',p_result.tolist())  
        #if maxp<p_result.max():
    #print('p_EM:',p_result)
    return list(p_result) 
  
#     for i_time in range(15):
#         for i in range(m):
#             for j in range(n):
#                 fenzi=0.0
#                 for kk in range(N):
#                     #kk=randint(0,N-1)
#                     fenzi+=get_bayes(att1data_rappor_list[kk], att2data_rappor_list[kk], att1signal_list, att2signal_list, p, f, i, j)
#                 p[i][j]=fenzi*1.0/N
#   
#         sump=0.0
#         for i in range(m):
#             for j in range(n):
#                 sump+=p[i][j]
#         for i in range(m):
#             for j in range(n):
#                 p_result[i][j]=p[m-1-i][n-1-j]/sump        
#         #print('deltap',deltap)    
#           
#         print(i_time,'p_EM:',p_result)
#     return list(p_result)
#######################################################################################################################################################################################################
def estimate_2d(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,bitsum_list,clique,f,dt=0.001):
    m=len(att1signal_list)
    n=len(att2signal_list)
    #print('cand1',att1signal_list)
    #print('cand2',att2signal_list)
    N=len(att1data_rappor_list)
    #p_cond=numpy.array([[0.0 for j in range(N)] for i in range(m*n)])
    p_cond=numpy.array([[[0.0 for j in range(N)] for i in range(n)] for k in range(m)])
    #p_cond=numpy.array([[0.0 for j in range(m*n)] for i in range(N)])
    p=numpy.array([[(1.0/(m*n)) for j in range(n)]for i in range(m)])
    #p=estimate_2d2(att1signal_list, att2signal_list, bitsum_list, clique)
    #p_result=[[0.0 for j in range(n)]for i in range(m)]
    p_result=numpy.array([[0.0 for j in range(n)]for i in range(m)])

    maxp=1.0 
    tt=0

    for kk in range(N):  
        for i in range(m):
            for j in range(n):
                    #print(att1signal_list)
                    #print(att2signal_list)
                if (p[i][j]>0.0):
                    p_cond[i][j][kk]=rappor_prob(att1data_rappor_list[kk], att1signal_list[i], f)*rappor_prob(att2data_rappor_list[kk], att2signal_list[j], f)                      
                #print (p_cond[i*(n)+j][kk])
    while(abs(p_result.max()-maxp)>0.001):
    #while(tt<=2):
        #print(tt,)
        tt+=1
        maxp=p_result.max()
        p_poster_temp=numpy.array([[0.0 for j in range(n)] for i in range(m)])
        for kk in range(N):
             
            fenmu=0.0
            fenzi=numpy.array([[0.0 for j in range(n)]for i in range(m)])
            
            for i2 in range(m):
                for j2 in range(n):
                    fenmu+=p[i2][j2]*p_cond[i2][j2][kk]
                             
                             
            for i in range(m):
                for j in range(n):
                    fenzi[i][j]=p[i][j]*p_cond[i][j][kk]
            
            for i in range(m):
                for j in range(n):
                    p_poster_temp[i][j]+=fenzi[i][j]/fenmu            
                     
        for i in range(m):
            for j in range(n):
                p[i][j]=p_poster_temp[i][j]/N
        
        #print('p',p)
        sump=0.0
        for i in range(m):
            for j in range(n):
                sump=sump+p[i][j]
        for i in range(m):
            for j in range(n):
                p_result[i][j]=p[i][j]/sump    
        #print('p_restult',p_result.tolist())  
        #if maxp<p_result.max():
    #print('p_EM:',p_result)
    return list(p_result) 

def estimate_2d6(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,bitsum_list,clique,f,dt=0.001):
    m=len(att1signal_list)
    n=len(att2signal_list)
    #print('cand1',att1signal_list)
    #print('cand2',att2signal_list)
    N=len(att1data_rappor_list)
    #p_cond=numpy.array([[0.0 for j in range(N)] for i in range(m*n)])
    p_cond=numpy.array([[[0.0 for j in range(N)] for i in range(n)] for k in range(m)])
    #p_cond=numpy.array([[0.0 for j in range(m*n)] for i in range(N)])
    #p=numpy.array([[(1.0/(m*n)) for j in range(n)]for i in range(m)])
    p=estimate_2d2(att1signal_list, att2signal_list, bitsum_list, clique)
    #p_result=[[0.0 for j in range(n)]for i in range(m)]
    p_result=numpy.array([[0.0 for j in range(n)]for i in range(m)])

    maxp=1.0 
    tt=0

    for kk in range(N):  
        for i in range(m):
            for j in range(n):
                    #print(att1signal_list)
                    #print(att2signal_list)
                if (p[i][j]>0.0):
                    p_cond[i][j][kk]=rappor_prob(att1data_rappor_list[kk], att1signal_list[i], f)*rappor_prob(att2data_rappor_list[kk], att2signal_list[j], f)                      
                #print (p_cond[i*(n)+j][kk])
    while(abs(p_result.max()-maxp)>0.001):
    #while(tt<=2):
        #print(tt,)
        tt+=1
        maxp=p_result.max()
        p_poster_temp=numpy.array([[0.0 for j in range(n)] for i in range(m)])
        for kk in range(N):
             
            fenmu=0.0
            fenzi=numpy.array([[0.0 for j in range(n)]for i in range(m)])
            
            for i2 in range(m):
                for j2 in range(n):
                    fenmu+=p[i2][j2]*p_cond[i2][j2][kk]
                             
                             
            for i in range(m):
                for j in range(n):
                    fenzi[i][j]=p[i][j]*p_cond[i][j][kk]
            
            for i in range(m):
                for j in range(n):
                    p_poster_temp[i][j]+=fenzi[i][j]/fenmu            
                     
        for i in range(m):
            for j in range(n):
                p[i][j]=p_poster_temp[i][j]/N
        
        #print('p',p)
        sump=0.0
        for i in range(m):
            for j in range(n):
                sump=sump+p[i][j]
        for i in range(m):
            for j in range(n):
                p_result[i][j]=p[i][j]/sump    
        #print('p_restult',p_result.tolist())  
        #if maxp<p_result.max():
    #print('p_EM:',p_result)
    return list(p_result)


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

# def rappor_list_paste(*attdata_rappor_list):   ####################################  for arbitrary lists
# 
#     attcombine_rappor_list=[]
#     for each_attdata_rappor_list in attdata_rappor_list:
#         attcombine_rappor_list.extend(each_attdata_rappor_list)
#         #temp_rappor_list=attcombine_rappor_list
#     return attcombine_rappor_list

def rappor_list_paste(*attdata_rappor_list):   ####################################  for arbitrary lists
    #print(attdata_rappor_list)
    #m=len(attdata_rappor_list[0])
    #attdata_rappor_list=list(attdata_rappor_list)
    m=len(attdata_rappor_list)
    n=len(attdata_rappor_list[0])
    #print(m,n)
    temp_rappor_list=[[]for i in range(n)]
    #temp_rappor_list=[[]]
    #print(attdata_rappor_list)
    for i in range(m):
        #print(temp_rappor_list)
       # print(attdata_rappor_list[i])
        attcombine_rappor_list=[x+y for x,y in zip(temp_rappor_list,attdata_rappor_list[i])]
        temp_rappor_list=attcombine_rappor_list
   # attcombine_rappor_list=map(list, zip(*attcombine_rappor_list))
    return attcombine_rappor_list

# a=[[1,0],[1,0],[1,1]]
# b=[[1,1],[1,0],[1,1]]
# print(rappor_list_paste(a,b))

def list_product(att1signal_list,att2signal_list):
    product_list=[]
    for att12_product in product(att1signal_list,att2signal_list):
        #print(att12_product)
        temp=[]
        for item in att12_product:
            temp.extend(item)
        #print(temp)
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

def row_product(att1signal_list,att2signal_list):
    product_list=[]
    for att12_product in product(att1signal_list,att2signal_list):
        #print(att12_product)
        #z=list(att12_product)
        product_list.append(list(att12_product))
    #print(product_list)
    return product_list

def row_list_paste(*attsignal_list):      ###################################### for arbitrary lists
    temp_row_list=[]
    for aproduct in product(*attsignal_list):
        temp_row_list.append(list(aproduct))
        
    return temp_row_list
def row_list_product(row_list,loc_list):   #this is the best way to create candidate set!!!!!!
    if len(loc_list)==1:
        att_row_list_combine=row_list[loc_list[0]]
    else:
        row_string=()
        for loc in loc_list:
            row_string=row_string+(row_list[loc],)
        att_row_list_combine=row_list_paste(*row_string)
    return att_row_list_combine

def att_combin(bit_list,bit_cand_list,row_list,loc_list):
    #loc_list specify the index of the attributes needed to be combined
    #print('********',row_list)
    
    if len(loc_list)==1:
        att_rappor_list_combine=bit_list[loc_list[0]]
        att_signal_list_combine=bit_cand_list[loc_list[0]]
        att_row_list_combine=row_list[loc_list[0]]
    else:
        rappor_para_string=()
        signal_para_string=()
        row_string=()
        for loc in loc_list:
            rappor_para_string=rappor_para_string+(bit_list[loc],)
            signal_para_string=signal_para_string+(bit_cand_list[loc],)
            #print('*************************',row_list[loc])
            row_string=row_string+(row_list[loc],)
            #print('**********row string',row_string)
            
        att_rappor_list_combine=rappor_list_paste(*rappor_para_string)
        att_signal_list_combine=signal_list_paste(*signal_para_string)
        att_row_list_combine=row_list_paste(*row_string)

    return att_rappor_list_combine,att_signal_list_combine,att_row_list_combine

def simple_combin(bit_list,row_list,loc_list):
    #loc_list specify the index of the attributes needed to be combined
    mulitilist=map(list, zip(*bit_list))
    #print(mulitilist[3])
    #multilist=bit_list
    selectlist=[]
    row_string=()
    for loc in loc_list:
        selectlist.append(mulitilist[loc])
        row_string=row_string+(row_list[loc],)   
    select_list=map(list,zip(*selectlist))
    att_row_list_combine=signal_list_paste(*row_string)
    str_select_loc=['|'.join(listeach) for listeach in select_list]
    str_candidate=['|'.join(listeach) for listeach in att_row_list_combine]
    return str_select_loc,str_candidate

def true_joint_distribution(multilist,rowlist,loclist):
    list1,list2=simple_combin(multilist, rowlist, loclist)
    #print('list2:',list2)
    freq_cat=Counter(list1)
    #print(freq_cat)
    freqnum=[]
    freqrate=[]
    for each_cat in list2:
        freqnum.append(freq_cat[each_cat])
    for i in range(len(freqnum)):
        if freqnum[i]==0:
            ff=0.0
        else:
            ff=1.0*freqnum[i]/sum(freqnum)
        freqrate.append(ff)
    #print(list)
        
    return list2,freqrate


def unfold_pro_list(pro):
    list(pro)
    unfold_pro=[]
    print(pro)
    for eachitem in pro:
        unfold_pro.extend(eachitem)
    return unfold_pro
    



#print(row_list_paste(['0','1'],['11','22'],['33','44']))
# for a in product(['0', '1'], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']):
#     print a



#file_id=4
# fai_C=0.2    #from 0.2, 0.3, 0.4, 0.5
#f=0.5   # from 0.1, 0.2, 0.3, 0.4, 0.5  *********
#bloombit=32
#hashbit=2
# dt=0.01
#readlimit=50000
#samplerate=0.001
#  
#att_num1,node_num1,true_node_num1,rowlist1,multilist1=Get_Params.get_file_info(file_id,readlimit,1.0)
#att_num,node_num,true_node_num,rowlist,multilist=Get_Params.get_file_info(file_id,readlimit,samplerate)
#bit_cand_list,bit_list,bitsum_list=Get_Rappor.rappor_process(bloombit, hashbit, f,att_num1,node_num1,true_node_num1,rowlist1,multilist1)
# print(bit_cand_list)
# print(bitsum_list)
# p_single=Get_Rappor.lasso_regression(bit_cand_list, bitsum_list)
# print(p_single)
# 
#pro=estimate_2d(att1data_rappor_list,att2data_rappor_list,att1signal_list,att2signal_list,bitsum_list,clique,f,dt=0.001)
# print(pro)
# testlist=[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
# testsum=[[121.0, 79.0, -89.0, -7.0, -91.0, 2927.0, -75.0, 33.0, -5.0, 51.0, 141.0, -57.0, 73.0, 49.0, 159.0, 151.0, 193.0, -237.0, 105.0, 21633.0, 203.0, 93.0, -25.0, 133.0, -17.0, 18413.0, 41.0, -93.0, -311.0, -151.0, 369.0, 17.0, -133.0, 153.0, 111.0, 35.0, -217.0, 4443.0, 195.0, 297.0, -303.0, -33.0, -1.0, 139.0, -115.0, -93.0, -103.0, -83.0, 87.0, 131.0, -153.0, 21809.0, -127.0, -9.0, 123.0, 41.0, -65.0, 16961.0, 33.0, 137.0, 163.0, 139.0, -41.0, 131.0]]
# p_test=lasso_regression(testlist,testsum)
# print(p_test)






############################################################################# Example for list process, i.e., listpaste, listproduct  
# 
# list1=[[1,2],[3,4]]
# list2=[[5,6],[7,8]]
# list3=[[9,10],[11,12]]
# z=[list1,list2,list3]
# print(z) 
# print(signal_list_paste(list1,list2,list3))
# print(list_paste(list1, list2,list2,list3))
# print(list_product(list1,list3))
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
#bit_cand_list,bit_list,bitsum_list=Get_Rappor.rappor_process(4, 8, 2, 0.01)
# att_num,node_num,rowlist,multilist=Get_Params.get_file_info(4)
# loc_list=[0,2,3,5,8]
# att_rappor_list_combine,att_signal_list_combine,att_row_list_combine=att_combin(bit_list, bit_cand_list, rowlist, loc_list)
# print(att_rappor_list_combine)
# print(att_signal_list_combine)
# print(att_row_list_combine)
######################################################################################################################################################################################################
# bit_list=map(list, zip(*bit_list))
# 
#p02=estimate_2d(bit_list[2], bit_list[2], bit_cand_list[2], bit_cand_list[2], 0.01, 0.001)
#att12data_rappor_list,att12signal_list=list_paste(bit_list[0], bit_list[2], bit_cand_list[0], bit_cand_list[2])
#   
#p02_3=estimate_2d(att12data_rappor_list, bit_list[3], att12signal_list, bit_cand_list[3], 0.01, 0.0001)
#print(p02)
# print(p02_3)
#p04=estimate_2d(bit_list[0], bit_list[0], bit_cand_list[0], bit_cand_list[0], 0.01, 0.0001)
#print(p04)

#lasso_cf=Get_Rappor.lasso_regression(bit_cand_list,bitsum_list)
#print(lasso_cf)
# att_num,node_num,rowlist,multilist=Get_Params.get_file_info(4, 40000, 0.01)
# loclist=[2,5,7,9,11]
# pro=true_joint_distribution(multilist, rowlist, loclist)
# print(pro)

        
    
    
    
