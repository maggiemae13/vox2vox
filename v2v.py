import numpy as np
import h5py
import pickle
from sklearn.linear_model import Ridge
import scipy.stats as sst
import pandas as pd
from tqdm import tqdm_notebook as tq
from matplotlib import pyplot as plt


def corr2_coeff(A,B):
    N = A.shape[0]
    # Colwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(0)[None,:]
    B_mB = B - B.mean(0)[None,:]

    # Sum of squares across cols
    ssA = (A_mA**2).sum(0) 
    ssB = (B_mB**2).sum(0) 
    ss = np.sqrt(ssA)*np.sqrt(ssB)
    # Finally get corr coeff
    return np.sum(A_mA*B_mB,axis=0)/ss


def save_stuff(save_to_this_file, data_objects_dict):
    failed = []
    with h5py.File(save_to_this_file+'.h5py', 'w') as hf:
        for k,v in data_objects_dict.iteritems():
            try:
                hf.create_dataset(k,data=v)
                #print 'saved %s in h5py file' %(k)
            except:
                failed.append(k)
                print 'failed to save %s as h5py. will try pickle' %(k)
    
    for k in failed:
        with open(save_to_this_file+'_'+'%s.pkl' %(k), 'w') as pkl:
            try:
                pickle.dump(data_objects_dict[k],pkl)
                #print 'saved %s as pkl' %(k)
            except:
                print 'failed to save %s in any format. lost.' %(k)


def choose_lambda(Trn_Source, Trn_Target, val_fraction, n_train_samples,nv, lambda_list):

    trnIdx = np.arange(0,n_train_samples)
    early_stop_num = np.round(len(trnIdx)*val_fraction).astype('int')
    perm_dx = np.random.permutation(trnIdx)
    validation_idx = perm_dx[0:early_stop_num]
    training_idx = perm_dx[early_stop_num:]
    
    
    Trn_Source_Val = Trn_Source[training_idx,:]
    Test_Source_Val = Trn_Source[validation_idx,:]
    
    if nv==1:
    
        Trn_Target_Val = Trn_Target[training_idx]
        Test_Target_Val = Trn_Target[validation_idx]
        score = np.zeros((len(lambda_list)))
        
    else:
        Trn_Target_Val = Trn_Target[training_idx,:]
        Test_Target_Val = Trn_Target[validation_idx,:]
        score = np.zeros((len(lambda_list),nv))
  
   

    for l, lamb in enumerate(lambda_list):
        regr = Ridge(alpha=lamb)
        regr.fit(Trn_Source_Val,Trn_Target_Val)
        Pval = regr.predict(Test_Source_Val)
        
        if nv==1:
            score[l] = sst.pearsonr(Test_Target_Val,Pval)[0]
        else:
            score[l, :] = corr2_coeff(Test_Target_Val,Pval)
    
    if nv==1:
        
        return lambda_list[score.argmax()]
    
    else:
    
        best_model = np.zeros(nv)

        for n in range(nv):
            best_model[n] = lambda_list[score[:,n].argmax()]


        return best_model



def vox2vox(source_list, target_list, train_data, test_data, 
            lambda_list, trial_type, val_fraction, roiNum, roi_mask,
            train_data2=None, test_data2=None, roi_mask2=None):
    

    if trial_type == 'average':
        nan_mask = np.isnan(np.concatenate((train_data, test_data))).sum(axis=0)==0

        if nan_mask.sum() > 0:
            train_data = train_data[:,nan_mask]
            test_data = test_data[:,nan_mask]
            roi_mask = roi_mask[nan_mask]
        
        
    elif trial_type == 'switch':
        
        assert train_data.shape[0]==3500, 'switch trial analysis only available for vim-1 data'
        assert test_data.shape[0]==1560, 'switch trial analysis only available for vim-1 data'
        
        nan_mask = np.isnan(np.concatenate((train_data, test_data))).sum(axis=0)==0

        if nan_mask.sum() > 0:
            train_data = train_data[:,nan_mask]
            test_data = test_data[:,nan_mask]      
            roi_mask = roi_mask[nan_mask]
        
        
    elif trial_type == 'cross':
        assert (train_data2 is not None) & (test_data2 is not None), 'Cross-Subject analysis requires second dataset'
        assert train_data.shape[0] == train_data2.shape[0], 'Train Data Samples not equal'
        assert test_data.shape[0] == test_data2.shape[0], 'Test Data Samples not equal'
        
        nan_mask = np.isnan(np.concatenate((train_data, test_data))).sum(axis=0)==0
        nan_mask2 = np.isnan(np.concatenate((train_data2, test_data2))).sum(axis=0)==0

        if nan_mask.sum() > 0:
            train_data = train_data[:,nan_mask]
            test_data = test_data[:,nan_mask]
            roi_mask = roi_mask[nan_mask]
        
        if nan_mask2.sum() > 0:
            train_data2 = train_data2[:,nan_mask2]
            test_data2 = test_data2[:,nan_mask2]
            roi_mask2 = roi_mask2[nan_mask2]

    else:
        raise Exception('unrecognized trial type')
    
    n_train_samples = train_data.shape[0]
    n_test_samples = test_data.shape[0]


    df_cc = pd.DataFrame(np.nan, index=source_list, columns=target_list,dtype=object)
    df_wts = pd.DataFrame(np.nan, index=source_list, columns=target_list,dtype=object)
    df_best_lambda = pd.DataFrame(np.nan, index=source_list, columns=target_list,dtype=object)

    for s, source in enumerate(tq(source_list)):
        print 'processing source roi ', source
        for t, target in enumerate(tq(target_list)):
            print  '    processing target roi ', target
            if trial_type in ['average', 'switch']:
                if source==target:
                    sn, tn = roiNum[source], roiNum[target]
                    df_cc.loc[source, target],df_wts.loc[source, target],df_best_lambda.loc[source, target] = train_lateral(sn , tn, train_data, test_data,n_train_samples, n_test_samples, lambda_list, trial_type,roi_mask, val_fraction, train_data2)

                else:
                    sn = roiNum[source]
                    tn = roiNum[target]
                    df_cc.loc[source, target],df_wts.loc[source, target], df_best_lambda.loc[source,target] = train_ff_fb(sn , tn, train_data, test_data,n_train_samples, n_test_samples, lambda_list, trial_type,roi_mask, val_fraction, train_data2)
                    
            else:
                sn = roiNum[source]
                tn = roiNum[target]
                df_cc.loc[source, target],df_wts.loc[source, target], df_best_lambda.loc[source,target] = train_ff_fb(sn , tn, train_data, test_data,n_train_samples, n_test_samples, lambda_list, trial_type,roi_mask, val_fraction,train_data2, test_data2, roi_mask2)


            
    return df_cc, df_wts, df_best_lambda


# In[2]:

def train_ff_fb(sn, tn, train_data, test_data, n_train_samples, n_test_samples, lambda_list,
                trial_type, roi_mask, val_fraction, train_data2=None, test_data2=None, roi_mask2=None):
    
    
    if trial_type=='average':
        
        Trn_Source = train_data[:, roi_mask==sn]
        Test_Source = test_data[:, roi_mask==sn]
        
        Trn_Target = train_data[:, roi_mask==tn]
        Test_Target = test_data[:, roi_mask==tn]
        
    elif trial_type=='cross':
    
        Trn_Source = train_data[:, roi_mask==sn]
        Test_Source = test_data[:, roi_mask==sn]
        
        Trn_Target = train_data2[:, roi_mask2==tn]
        Test_Target = test_data2[:, roi_mask2==tn]
        
    
    elif trial_type=='switch':
        
        Trn_Source = train_data[:, roi_mask==sn]
        Test_Source = test_data[:, roi_mask==sn]
        
        Trn_Target =  np.concatenate([train_data[1750:,:],train_data[:1750,:]])
        Trn_Target = Trn_Target[:, roi_mask==tn]
        Test_Target = test_data[:, roi_mask==tn]
        
    nv = Test_Target.shape[1]    
    best_lambda = choose_lambda(Trn_Source, Trn_Target, val_fraction, n_train_samples,nv,lambda_list)
    
    regr = Ridge(alpha=best_lambda)
    regr.fit(Trn_Source, Trn_Target)
    
    
    if trial_type in ['average', 'cross']:
        
        predictions = regr.predict(Test_Source)
        return corr2_coeff(Test_Target, predictions), regr.coef_, best_lambda
    
    else:
        
        
        val_cc = np.zeros((nv,13))
        val_cc_own = np.zeros((nv,13))
        
        for r in range(13):
            test_reps = range(120*r,120*(r+1))
            test_src = np.tile(Test_Source[test_reps,:],[12,1])

            predictions = regr.predict(test_src)

            val_cc[:,r] = corr2_coeff(np.delete(Test_Target,test_reps, axis=0),predictions)
            val_cc_own[:,r] = corr2_coeff(Test_Target[test_reps,:],predictions[:120,:])
                
        return np.concatenate([val_cc,val_cc_own],axis=1), regr.coef_, best_lambda
        


# In[3]:

def train_lateral(sn, tn, train_data, test_data, n_train_samples, n_test_samples, lambda_list,
                trial_type, roi_mask, val_fraction, train_data2=None, test_data2=None, roi_mask2=None):

    
    if trial_type=='average':
        
        Trn_Source = train_data[:, roi_mask==sn]
        Test_Source = test_data[:, roi_mask==sn]
        
        Trn_Target = train_data[:, roi_mask==tn]
        Test_Target = test_data[:, roi_mask==tn]
        
    elif trial_type=='cross':
    
        Trn_Source = train_data[:, roi_mask==sn]
        Test_Source = test_data[:, roi_mask==sn]
        
        Trn_Target = train_data2[:, roi_mask2==tn]
        Test_Target = test_data2[:, roi_mask2==tn]
        
    elif trial_type=='switch':
        
        Trn_Source = train_data[:, roi_mask==sn]
        Test_Source = test_data[:, roi_mask==sn]
        
        Trn_Target =  np.concatenate([train_data[1750:,:],train_data[:1750,:]])
        Trn_Target = Trn_Target[:, roi_mask==tn]
        Test_Target = test_data[:, roi_mask==tn]
    
    
    nv = Test_Target.shape[1] 
    
    if trial_type == 'average':
        val_cc = np.empty(nv)
        val_wts = np.empty((nv,nv-1))
        best_lambs = np.empty(nv)
        
        for v in range(nv):

            Trn_Source_l = np.delete(Trn_Source, v, axis=1)
            Test_Source_l = np.delete(Test_Source, v, axis=1)

            Trn_Target_l = Trn_Target[:,v]
            Test_Target_l = Test_Target[:,v]

            best_lambda = choose_lambda(Trn_Source_l, Trn_Target_l, val_fraction, n_train_samples,1,lambda_list)

            regr = Ridge(alpha=best_lambda)
            regr.fit(Trn_Source_l, Trn_Target_l)
            Pval = regr.predict(Test_Source_l)
            val_cc[v] = sst.pearsonr(Test_Target_l,Pval)[0]
            val_wts[v,:] = regr.coef_
            best_lambs[v]=best_lambda
        
        return val_cc, val_wts, best_lambs
    
    else:
        
        val_cc = np.zeros((nv,13))
        val_cc_own = np.zeros((nv,13))
        val_wts = np.empty((nv,nv-1))
        best_lambs = np.empty(nv)
        
        for v in range(nv):
            
            Trn_Source_l = np.delete(Trn_Source, v, axis=1)
            Test_Source_l = np.delete(Test_Source, v, axis=1)

            Trn_Target_l = Trn_Target[:,v]
            Test_Target_l = Test_Target[:,v]

            best_lambda = choose_lambda(Trn_Source_l, Trn_Target_l, val_fraction, n_train_samples,1,lambda_list)

            regr = Ridge(alpha=best_lambda)
            regr.fit(Trn_Source_l, Trn_Target_l)

            val_wts[v,:] = regr.coef_
            best_lambs[v]=best_lambda
        
            for r in range(13):
                test_reps = range(120*r,120*(r+1))
                test_src = np.tile(Test_Source_l[test_reps,:],[12,1])

                predictions = regr.predict(test_src)

                val_cc[v,r] = sst.pearsonr(np.delete(Test_Target_l,test_reps, axis=0),predictions)[0]
                val_cc_own[v,r] = sst.pearsonr(Test_Target_l[test_reps],predictions[:120])[0]
                
        return np.concatenate([val_cc,val_cc_own],axis=1), regr.coef_, best_lambda
        

def eccentricity(areas, fwrf_cc, fwrf_xy, v2v_wts, roiNum, roimask, x_range,y_range, th=.21,bins=20):
            
    # this function calculates receptive field center based upon vox2vox models
    # previously this function used hexbin. now we are using numpy 2dhist function
    """
    Inputs:
    
    <areas> list of source/target areas to compute v2v rfs for
    <fwrf_cc> prediction accuracy values for each voxel in the fwrf model
    <fwrf_xy> receptive field center values from fwrf model [N voxels, x, y]
    <v2v_wts> dataframe with weights from vox2vox model for each source/target pairing
    <roiNum> mapping of area names to number value in roimask
    <roimask> same shape as fwrf_cc with area values for each voxel
    <th> threshold value to select voxels from fwrf to use in receptive field calculation. 
        default .21 corresponds to tpreviously computed hreshold for averaged betas
    <x_range> list of v2v x center values to choose from (should be same as fwrf vals)
    <y_range> list of v2v y center values to choose from (should be same as fwrf vals)
    <bins> how many bins to use in 2dhist. Should be the number of center values in a direction.
    Returns:
    
    <df_v2v_xy> dataframe with v2v receptive field centers for each voxel in each source/target pairing. 
    
    """

    df_v2v_xy = pd.DataFrame(np.nan, index=areas, columns=areas,dtype=object)
    tq.monitor_interval = 0 # this is necessary to stop some weird twdm error

    for s, src in enumerate(areas):
        print 'processing source ', src
        
        for t, trg in enumerate(areas):
            print 'processing target ', trg

            x = roiNum[src]; y = roiNum[trg];            

            v2v_xy = np.zeros((len(fwrf_xy[roimask==y,0]),2))
            nv = len(fwrf_xy[roimask==y,0])

            for i in tq(range(nv)):

                if src == trg:
                    w = v2v_wts.loc[src,trg][i,:][np.delete(fwrf_cc[roimask==x],i)>=th]
                    src_x = np.delete(fwrf_xy[roimask==x,0],i)[np.delete(fwrf_cc[roimask==x],i)>=th]
                    src_y = np.delete(fwrf_xy[roimask==x,1],i)[np.delete(fwrf_cc[roimask==x],i)>=th]
                else:
                    w = v2v_wts.loc[src,trg][i,:][fwrf_cc[roimask==x]>=th]
                    src_x = fwrf_xy[roimask==x,0][fwrf_cc[roimask==x]>=th]
                    src_y = fwrf_xy[roimask==x,1][fwrf_cc[roimask==x]>=th] 
                
                xc,yc=np.unravel_index(np.histogram2d(src_x,src_y,bins=bins, weights=w)[0].argmax(), [bins,bins])

                v2v_xy[i,0] = x_range[xc]
                v2v_xy[i,1] = y_range[yc]
          
            df_v2v_xy.loc[src, trg] = v2v_xy
            
    return df_v2v_xy