
import os
from os.path import exists
import torch

# to read .mat files
import mat4py

# to read .pkl files
from all_imports import *
import pandas as pd

# to do math operations
import numpy as np
import scipy.io as io

# to find peaks in signal
from scipy import signal as signal

# To use the network
import uneye

# to get list of files
from glob import glob as dir

# to do dimensionality reduction

import umap

# to do unsupervised clustering
#from sklearn.cluster import DBSCAN
import hdbscan

# to plot things
from matplotlib import pyplot as plt 
from matplotlib.pyplot import *
matplotlib.style.use('classic')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def detect_CS(weights_name, filename = None, LFP = None, High_passed = None ,output_name = 'same',extention = 'same',plot = False, plot_only_good = True, save = True, sampling_frequency = 25000, ks=9,mp=9, realign = True, cluster = True,alignment_w = (-.5,2),cluster_w = (-2,2),plot_w= (-4,8)):
    # important aguments:
    # -filename is the filename path. If it is not defined then you should input the LFP and the High-passed signal
    # -weights_name is the path of the weight or just the name of the weight if it is stored in /training
    # -sampling_frequency is the sampling frequency of the signal analysed
    # ks and mp are the size of the kernel and max pooling operations respectively. 
    if plot:
        cmap = plt.get_cmap('jet')

    
    samp  = int(sampling_frequency/1000) # Khz
    if filename != None:
        if output_name == 'same':
            output_name = filename[:-4]
            extention = '_predicted_CS_time_from_LFP_and_volt' # You can modify this
        # load data
        data = mat4py.loadmat(filename)
        try:
            # High passed signal
            High_passed = data['Data']['PC_Volt'] 
        except:
            # High passed signal
            High_passed = data['Data']['PC_Volt_Trace']    
        # LFP signal
        LFP = data['Data']['LFP']
        
        High_passed = np.asarray(High_passed)
        LFP = np.asarray(LFP)
    else:
        High_passed = High_passed 
        LFP = LFP
    if len(LFP)==0 or len(High_passed)==0:
        labels = {'cs_onset': [],
                   'cs_offset': [],
                   'cluster_ID': [],
                 'embedding': []}
        if save == True:
            print('saving '+output_name+extention+'.mat')
            io.savemat(output_name+extention+'.mat',labels)
        return([],[],[],[])
    
    trial_length = 1 #sec, length per "trial"
    trial_length *= samp*1000
    if np.max(LFP.shape)>trial_length:
        # preprocessing: cut recording into overlapping pieces because the network can't deal with really big signals
        total_length = np.max(LFP.shape) #total recording length
        
        overlap = 0.1 #sec, length of overlap
        overlap *= samp*1000
        if np.floor(total_length/(trial_length-overlap))<(total_length-overlap)/(trial_length-overlap):
            num_steps = int(np.floor(total_length/(trial_length-overlap)))
        else:
            num_steps = int(np.floor(total_length/(trial_length-overlap)))-1
        steps = np.array([i*trial_length-i*overlap for i in range(num_steps)]).astype(int)
        LFP_mat = np.zeros((num_steps+1,trial_length))
        High_mat = np.zeros((num_steps+1,trial_length))
        
        for i,s in enumerate(steps):
            LFP_mat[i,:] = LFP[s:s+int(trial_length)]
            High_mat[i,:] = High_passed[s:s+int(trial_length)]
            
        # exception for last piece (from end-trial_length to end)
        LFP_mat[-1,:] = LFP[-int(trial_length):]
        High_mat[-1,:] = High_passed[-int(trial_length):]
        
        # normalise LFP as a multiple of absolute median
        for i in np.arange(0,np.shape(LFP_mat)[0]):
            LFP_mat[i,:] = LFP_mat[i,:]-np.median(LFP_mat[i,:])
            LFP_mat[i,:] = LFP_mat[i,:]/np.median(np.abs(LFP_mat[i,:]))
    else:
        LFP_mat = LFP
        High_mat = High_passed
    # U'n'Eye
    model = uneye.DNN(ks=ks,mp=mp,classes=2,
                  weights_name=weights_name,sampfreq=sampling_frequency,min_sacc_dur=1)
    Pred,Prob = model.predict(LFP_mat,High_mat)
    
    if np.max(LFP.shape)>trial_length:
        # recover original shape
        Prediction = np.zeros(total_length)
        Probability = np.zeros(total_length)
        for i,s in enumerate(steps):
            if i==0:
                Prediction[s:s+int(trial_length)-int(overlap/2)] = Pred[i,:-int(overlap/2)]
                Probability[s:s+int(trial_length)-int(overlap/2)] = Prob[i,1,:-int(overlap/2)]
            else:
                Prediction[s+int(overlap/2):s+int(trial_length)-int(overlap/2)] = Pred[i,int(overlap/2):-int(overlap/2)]
                Probability[s+int(overlap/2):s+int(trial_length)-int(overlap/2)] = Prob[i,1,int(overlap/2):-int(overlap/2)]
        # exception for last piece
        Prediction[-int(trial_length)+int(overlap/2):] = Pred[-1,int(overlap/2):]
        Probability[-int(trial_length)+int(overlap/2):] = Prob[-1,1,int(overlap/2):]
    else:
        Prediction = Pred
        Probability = Prob
    cs_onset=np.argwhere(np.diff(Prediction)==1);
    cs_offset=np.argwhere(np.diff(Prediction)==-1);
    
    if cluster ==False & realign==False: # stop early if everything that is needed is the raw output from the network
        labels = {'cs_onset':cs_onset,
                   'cs_offset':cs_offset}
        if save == True:
            print('saving '+output_name+extention+'.mat')
            io.savemat(output_name+extention+'.mat',labels)
        return(cs_onset,cs_offset)
    
    alignment_window = (np.array([sampling_frequency*alignment_w[0],sampling_frequency*alignment_w[1]])/1000).astype(int) # 2 ms to realign CS onset
    cluster_window = (np.array([sampling_frequency*cluster_w[0],sampling_frequency*cluster_w[1]])/1000).astype(int)  # consider the first 2 ms after CS onset for clustering
    plot_window = (np.array([sampling_frequency*plot_w[0],sampling_frequency*plot_w[1]])/1000).astype(int) # plot between 4 ms before and 8 ms after CS onset
    
    # remove CS detected to close from the edges of the signal
    cs_offset = cs_offset[(cs_onset>sampling_frequency*(-plot_w[0])/1000)& (cs_onset<len(Prediction)-sampling_frequency*plot_w[1]/1000)]
    cs_onset = cs_onset[(cs_onset>sampling_frequency*(-plot_w[0])/1000)& (cs_onset<len(Prediction)-sampling_frequency*plot_w[1]/1000)]

    if len(cs_onset)>0:
        ######################################## 
        # post processing
        ########################################
        
        

        # align each complex spike to the overall average
        
        average_CS = np.zeros((len(cs_onset),alignment_window[1]-alignment_window[0]))
        for i,j in enumerate(cs_onset):
            average_CS[i,:]=High_passed[j+alignment_window[0]:j+alignment_window[1]]
        
        norm_signal = np.median(np.abs(High_passed))/0.6745
        corrected_on=np.zeros((len(cs_onset),1))
        
        if realign:
            for i,j in enumerate(cs_onset):
                
                c=np.correlate(np.mean(average_CS,axis=0),average_CS[i,:],"full")
                lag = (np.argmax(c)-c.size/2)+.5
                corrected_on[i]=int(j-lag)
        else:     
            corrected_on = cs_onset
          
        if cluster == False: # if only the realignment of onsets was needed
            cs_onset = corrected_on
            labels = {'cs_onset':cs_onset,
                     'cs_offset':cs_offset}
            if save == True:
                print('saving '+output_name+extention+'.mat')
                io.savemat(output_name+extention+'.mat',labels)  
            return(cs_onset,cs_offset)
        
        average_CS2 = np.zeros((len(cs_onset),cluster_window[1]-cluster_window[0])) 
        for i,j in enumerate(corrected_on):
            ind = int(j)
            average_CS2[i,:] = High_passed[ind+cluster_window[0]:ind+cluster_window[1]]
        
        average_prob = np.zeros((len(cs_onset),plot_window[1]-plot_window[0]))
        for i,j in enumerate(corrected_on):
            ind = int(j)
            average_prob[i,:] = Probability[ind+plot_window[0]:ind+plot_window[1]]
            
        average_LFP = np.zeros((len(cs_onset),cluster_window[1]-cluster_window[0]))
        for i,j in enumerate(corrected_on):
            ind = int(j)
            average_LFP[i,:] = LFP[ind+cluster_window[0]:ind+cluster_window[1]]
            
        if plot:
            average_CS_plot = np.zeros((len(cs_onset),plot_window[1]-plot_window[0]))
            for i,j in enumerate(cs_onset):
                ind = int(j)
                average_CS_plot[i,:]=High_passed[ind+plot_window[0]:ind+plot_window[1]]
            average_CS2_plot = np.zeros((len(cs_onset),plot_window[1]-plot_window[0]))
            for i,j in enumerate(corrected_on):
                ind = int(j)
                average_CS2_plot[i,:]=High_passed[ind+plot_window[0]:ind+plot_window[1]]
            average_LFP_plot = np.zeros((len(cs_onset),plot_window[1]-plot_window[0]))
            for i,j in enumerate(corrected_on):
                ind = int(j)
                average_LFP_plot[i,:] = LFP[ind+plot_window[0]:ind+plot_window[1]]
            
            
            u = np.random.permutation(i)[:np.round(i/2).astype(int)]
            fig = plt.figure()
            ax1 = plt.subplot(411)
            ax2 = plt.subplot(412)
            for z in u:
                ax1.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,average_CS_plot[z,:],c = 'k',Linewidth = 0.1)
                ax2.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,average_CS2_plot[z,:],c = 'k',Linewidth = 0.1)
            plt.show()
            fig.savefig('realignment.pdf')
        # dimensionality reduction using UMAP
        n_neighbors = 15
        if np.shape(average_CS2)[0]<n_neighbors: # exit if the neumber of complex spike detected is too small
          labels = {'cs_onset': [],
                   'cs_offset': [],
                   'cluster_ID': [],
                 'embedding': []}
          if save == True:
            print('saving '+output_name+extention+'.mat')
            io.savemat(output_name+extention+'.mat',labels)
            return([],[],[],[])
            return
        embedding = umap.UMAP(n_neighbors = n_neighbors , min_dist=0.00001) 
        embedding = embedding.fit_transform(np.concatenate((average_CS2, average_LFP), axis=1))
    
        # clustering
        clustering = hdbscan.HDBSCAN(allow_single_cluster=True)
        clustering = clustering.fit(embedding)
        
        
        # keep all cluster of waveforms for which the average predictive probability lasts at least 3 ms
          
        tree = clustering.single_linkage_tree_.to_pandas()
        penultimate = np.sort(tree['size'])[-3]
        last = np.sort(tree['size'])[-2]
        distance =np.concatenate((np.array(tree['distance'][tree['size']==penultimate]), np.array(tree['distance'][tree['size']==last])), axis=None).mean()
        labels_big = clustering.single_linkage_tree_.get_clusters(distance,min_cluster_size=0)
        
        
        include = np.zeros((1,len(cs_onset))).astype(bool)[0]
        good_CS = []
        if plot:
            colors = cmap(np.linspace(0, 1.0, len(np.unique(labels_big))))
            fig = plt.figure()
            ax0 = plt.subplot(311)
            ax1 = plt.subplot(312)
            ax2 = plt.subplot(313)
        for i,lab in enumerate(np.unique(labels_big)):
            good_CS.append((sum((np.mean(average_prob[labels_big==lab,:],axis = 0)>.5).astype(float))/samp)>3)
            if good_CS[i]:
                include[labels_big==lab] = True
            if plot:
                if plot_only_good == True:
                    if good_CS[i]:
                        u = np.argwhere(labels_big==lab)
                        u = np.random.permutation(u)[:np.round(len(u)/2).astype(int)]
                        for z in u:
                            ax1.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,average_CS2_plot[z[0],:],c = colors[i], Linewidth = 0.1)
                            ax0.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,average_LFP_plot[z[0],:],c = colors[i], Linewidth = 0.1)
                        
                        ax0.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,np.mean(average_LFP_plot[labels_big==lab,:],axis = 0),c = colors[i])
                        ax1.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,np.mean(average_CS2_plot[labels_big==lab,:],axis = 0),c = colors[i])
                        ax2.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,np.mean(average_prob[labels_big==lab,:],axis = 0), c = colors[i])
                else:
                    u = np.argwhere(labels_big==lab)
                    u = np.random.permutation(u)[:np.round(len(u)/2).astype(int)]
                    for z in u:
                        ax1.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,average_CS2_plot[z[0],:],c = colors[i],Linewidth = 0.1)
                        ax0.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,average_LFP_plot[z[0],:],c = colors[i],Linewidth = 0.1)
                    
                    ax0.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,np.mean(average_LFP_plot[labels_big==lab,:],axis = 0),c = colors[i])
                    ax1.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,np.mean(average_CS2_plot[labels_big==lab,:],axis = 0),c = colors[i])
                    ax2.plot(np.arange(plot_window[0],plot_window[1])/sampling_frequency*1000,np.mean(average_prob[labels_big==lab,:],axis = 0),c = colors[i])
                      
        if plot:    
            xlabel('time from CS onset (ms)',fontsize=15)
            ylabel('Voltage (uV)',fontsize=15)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_facecolor('none')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_facecolor('none')
            ax2.axis(ymin=0,ymax=1.01)
            plt.show()
            fig.savefig('example.pdf')
        
        fig = plt.figure()
        for i,lab in enumerate(np.unique(labels_big)):
            if plot:
                if plot_only_good == True:
                    if good_CS[i]:
                        plt.scatter(embedding[labels_big==lab,0],embedding[labels_big==lab,1],c = colors[i], edgecolors = 'face')
                else:
                    plt.scatter(embedding[labels_big==lab,0],embedding[labels_big==lab,1],c = colors[i], edgecolors = 'face')
        fig.savefig('dimensionality_reduction.pdf')
        cs_onset = corrected_on[include];
        cs_offset = cs_offset[include]
        
        cs_onset = cs_onset.astype('int')
        cs_offset = cs_offset.astype('int')
    
        labels = {'cs_onset':cs_onset,
                   'cs_offset':cs_offset,
                   'cluster_ID': labels_big[include],
                 'embedding': embedding[include,:]}
        if save == True:
            print('saving '+output_name+extention+'.mat')
            io.savemat(output_name+extention+'.mat',labels)
        return(cs_onset,cs_offset,labels_big[include],embedding[include,:])
    else:
        labels = {'cs_onset': [],
                   'cs_offset': [],
                   'cluster_ID': [],
                 'embedding': []}
        if save == True:
            print('saving '+output_name+extention+'.mat')
            io.savemat(output_name+extention+'.mat',labels)
        return([],[],[],[])
    
    