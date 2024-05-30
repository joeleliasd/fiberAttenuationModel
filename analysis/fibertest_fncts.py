#print("gotcha bitch")
#exit()

#this python script contains all unpacking, fitting, and plotting function definitions
#some dependencies:
import numpy as np
import pandas as pd #this might need to be installed
import datetime
import matplotlib.pyplot as plt
import statistics as sts
from scipy.optimize import minimize,least_squares

frdmunits=2.54 #cm/inch
pos=np.transpose(np.array(pd.read_csv("fibertest_000915.csv")))[0]*frdmunits

# Data Unpacker, reads .csv file and runs it through an algorithm that throws out outliers (+/- 2*sigma from mean),
# and produces new a 4x25 data file of pmt_avg, pmt_std, sipm_svg, sipm_std
def unpack_data(datafile,npos,size):

    data_list=np.transpose(np.array(pd.read_csv(datafile)))

    points=int((len(data_list)-6)/2) # This is the number of data points taken
    #size is the number of data points we want to unpack
    newpmtstd=np.empty(npos)
    newpmtmean=np.empty(npos)
    newsipmstd=np.empty(npos)
    newsipmmean=np.empty(npos)

    npmtoutlier=0
    nsipmoutlier=0

    for i in range(npos):
        np.transpose(data_list)[i][2:-(2*points-size+4)]
        stdev_pmt=sts.stdev(np.transpose(data_list)[i][2:-(2*points-size+4)])
        avg_pmt=sts.mean(np.transpose(data_list)[i][2:-(2*points-size+4)])
        newpmtdata=[]

        np.transpose(data_list)[i][-(points+4):-(4+(points-size))]
        stdev_sipm=sts.stdev(np.transpose(data_list)[i][-(points+4):-(4+(points-size))])
        avg_sipm=sts.mean(np.transpose(data_list)[i][-(points+4):-(4+(points-size))])
        newsipmdata=[]


        for j in range(len(np.transpose(data_list)[i][2:-(points-size+4)])):
            if np.transpose(data_list)[i][j+2]<avg_pmt+2*stdev_pmt and np.transpose(data_list)[i][j+2]>avg_pmt-2*stdev_pmt:
                newpmtdata.append(np.transpose(data_list)[i][j+2])
                #print("good")
            else:
                npmtoutlier+=1

        for j in range(len(np.transpose(data_list)[i][-(points+4):-(4+(points-size))])):
            if np.transpose(data_list)[i][j+points+2]<avg_sipm+2*stdev_sipm and np.transpose(data_list)[i][j+points+2]>avg_sipm-2*stdev_sipm:
                newsipmdata.append(np.transpose(data_list)[i][j+points+2])
                #print("good")
            else:
                nsipmoutlier+=1

        newpmtstd[i]=sts.stdev(newpmtdata)*10**6
        newpmtmean[i]=-sts.mean(newpmtdata)*10**6
        newsipmstd[i]=sts.stdev(newsipmdata)*10**6
        newsipmmean[i]=-sts.mean(newsipmdata)*10**6
        #print(size-len(newpmtdata))
    return np.array([newpmtmean,newpmtstd,newsipmmean,newsipmstd])

def unpack_and_correct(filename):
    pmt_data=np.transpose(np.array(pd.read_csv(filename)))[2:-204]
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-204:-4]
    sipm_mean=np.mean(sipm_data)
    sipm_stddev=sts.stdev(np.ndarray.flatten(sipm_data))

    nsigma=2

    lsr_crctd_data_stdev=np.empty(25)
    lsr_crctd_data_mean=np.empty(25)
    for mpos in range(0,25):
        new_data=[]
        ndata=0
        pmt_mean=np.mean(np.transpose(pmt_data)[mpos])
        pmt_stddev=sts.stdev(np.transpose(pmt_data)[mpos])

        for run in range(0,200):

            if sipm_data[run][mpos]<(sipm_mean+sipm_stddev) and sipm_data[run][mpos]>(sipm_mean-sipm_stddev) and pmt_data[run][mpos]<(pmt_mean+nsigma*pmt_stddev) and pmt_data[run][mpos]>(pmt_mean-nsigma*pmt_stddev):
                #print(run)
                new_data.append(pmt_data[run][mpos])
        #print(len(new_data))
        #print(npos)
        lsr_crctd_data_stdev[mpos]=-sts.stdev(new_data)*10**6
        lsr_crctd_data_mean[mpos]=-sts.mean(new_data)*10**6
    #fig = plt.figure(figsize=(16, 10), dpi=80)
    #plt.errorbar(pos,lsr_crctd_data_mean, yerr=lsr_crctd_data_stdev, fmt="o", color="k",capsize=5,label='Fiber 2 Data')
    #plt.plot()
    return lsr_crctd_data_mean,lsr_crctd_data_stdev

def unpack_and_correct_2(filename):
    pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data=np.transpose(np.array(pd.read_csv(filename)))[2:-204] # all data taken by the pmt per position
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-204:-4] # all data taken by the sipm per position

    nsigma=1 # number of standard deviations to include 

    lsr_crctd_data_stdev=np.empty(25) # empty lists to store corrected standard deviation per position
    lsr_crctd_data_mean=np.empty(25) # empty lists to store corrected mean per position
    for mpos in range(0,25):
        new_data=[]
        ndata=0
        
        pmt_mean=np.mean(np.transpose(pmt_data)[mpos])
        pmt_stddev=sts.stdev(np.transpose(pmt_data)[mpos])
        
        sipm_mean=np.mean(np.transpose(sipm_data)[mpos])
        sipm_stddev=sts.stdev(np.transpose(sipm_data)[mpos])

        for run in range(len(pmt_data)):
            a,b=np.polyfit(np.transpose(sipm_data)[mpos], np.transpose(pmt_data)[mpos], deg=1)
            fit_mean=sts.mean(a*np.transpose(sipm_data)[mpos]+b)
            fit_std=sts.stdev(a*np.transpose(sipm_data)[mpos]+b)

            if pmt_data[run][mpos]<fit_mean+3*fit_std and pmt_data[run][mpos]>fit_mean-3*fit_std:
                new_data.append(pmt_data[run][mpos])

        #for run in range(0,len(pmt_data)):

        #    if sipm_data[run][mpos]<(sipm_mean+sipm_stddev) and sipm_data[run][mpos]>(sipm_mean-sipm_stddev) and pmt_data[run][mpos]<(pmt_mean+nsigma*pmt_stddev) and pmt_data[run][mpos]>(pmt_mean-nsigma*pmt_stddev):
                #print(run)
        #        new_data.append(pmt_data[run][mpos])
        #print(len(new_data))
        #print(npos)
        lsr_crctd_data_stdev[mpos]=sts.stdev(new_data)*10**6
        lsr_crctd_data_mean[mpos]=-sts.mean(new_data)*10**6
    fig = plt.figure(figsize=(16, 10), dpi=80)
    plt.errorbar(pos,lsr_crctd_data_mean, yerr=lsr_crctd_data_stdev, fmt="o", color="k",capsize=5,label='Fiber 2 Corrected')
    plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber 2')
    plt.legend()
    plt.plot()
    return lsr_crctd_data_mean,lsr_crctd_data_stdev
    
def unpack_and_correct_by_width(filename,runs):
    pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data=np.transpose(np.array(pd.read_csv(filename)))[2:-(runs+4)] # all data taken by the pmt per position
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-(runs+4):-4] # all data taken by the sipm per position
    f2_nocut_si_filtered=[]
    for i in range(25):
        data=np.transpose(sipm_data)[i]
        # Create a box plot
        box_props = plt.boxplot(data)
        
       # plt.show()

        # Get the whisker ranges
        lower_whisker = box_props['whiskers'][0].get_ydata()[1]
        upper_whisker = box_props['whiskers'][1].get_ydata()[1]

        # Identify outliers using IQR method
        iqr = upper_whisker - lower_whisker
        lower_bound = lower_whisker #- 2 * iqr
        upper_bound = upper_whisker #+ 2 * iqr

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        #print(outliers,lower_bound,upper_bound)
    
         # Remove outliers from the data
        filtered_data = np.delete(data, outliers)
        f2_nocut_si_filtered.append(filtered_data)
    #sipm_mean=np.mean(sipm_data)
    #sipm_stddev=np.std(sipm_data)

    nsigma=1 # number of standard deviations to include 

    lsr_crctd_data_stdev=np.empty(25) # empty lists to store corrected standard deviation per position
    lsr_crctd_data_mean=np.empty(25) # empty lists to store corrected mean per position
    lsr_crctd_data_mean_si=np.empty(25)
    lsr_crctd_data_stdev_si=np.empty(25)
    for mpos in range(0,25):
        new_data=[]
        new_data_sipm=[]
        ndata=0
        sipm_mean=np.mean(f2_nocut_si_filtered[mpos])
        sipm_stddev=np.std(f2_nocut_si_filtered[mpos])
        pmt_mean=np.mean(np.transpose(pmt_data)[mpos])
        pmt_stddev=np.std(np.transpose(pmt_data)[mpos])

        for run in range(len(pmt_data)):
            pmt_val=pmt_data[run][mpos]
            sipm_val=sipm_data[run][mpos]
            if pmt_val<pmt_mean+0.5*pmt_stddev and pmt_val>pmt_mean-0.5*pmt_stddev and sipm_val<sipm_mean+0.5*sipm_stddev and sipm_val>sipm_mean-0.5*sipm_stddev:
                new_data.append(pmt_val)
                new_data_sipm.append(sipm_val)

        lsr_crctd_data_stdev[mpos]=np.std(new_data)*10**6
        lsr_crctd_data_mean[mpos]=-np.mean(new_data)*10**6
        lsr_crctd_data_mean_si[mpos]=np.mean(new_data_sipm)*10**6
        lsr_crctd_data_stdev_si[mpos]=np.std(new_data_sipm)*10**6
        
    fig = plt.figure(figsize=(16, 10), dpi=80)
    # Error propagation formula for the ratio
    #err = (lsr_crctd_data_mean/pmt_mean_old) * np.sqrt((lsr_crctd_data_stdev / lsr_crctd_data_mean)**2 + (pmt_std_old / pmt_mean_old)**2)
    err = (lsr_crctd_data_stdev/pmt_mean_old)
    plt.errorbar(pos,lsr_crctd_data_mean/pmt_mean_old, yerr=lsr_crctd_data_stdev, fmt="o", color="k",capsize=5,label='Ratio of corrected to raw data')
    #plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber')
    plt.legend()
    plt.show()
    return lsr_crctd_data_mean,lsr_crctd_data_stdev,lsr_crctd_data_mean_si,lsr_crctd_data_stdev_si

def unpack_and_correct_by_width_trbsh(filename):
    pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data=np.transpose(np.array(pd.read_csv(filename)))[2:-204] # all data taken by the pmt per position
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-204:-4] # all data taken by the sipm per position
    
    f2_nocut_si_filtered=[]
    for i in range(25):
        data=np.transpose(sipm_data)[i]
        # Create a box plot
        box_props = plt.boxplot(data)
        
       # plt.show()

        # Get the whisker ranges
        lower_whisker = box_props['whiskers'][0].get_ydata()[1]
        upper_whisker = box_props['whiskers'][1].get_ydata()[1]

        # Identify outliers using IQR method
        iqr = upper_whisker - lower_whisker
        lower_bound = lower_whisker #- 2 * iqr
        upper_bound = upper_whisker #+ 2 * iqr

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        #print(outliers,lower_bound,upper_bound)
    
         # Remove outliers from the data
        filtered_data = np.delete(data, outliers)
        f2_nocut_si_filtered.append(filtered_data)
    #sipm_mean=np.mean(sipm_data)
    #sipm_stddev=np.std(sipm_data)

    nsigma=1 # number of standard deviations to include 

    lsr_crctd_data_stdev=np.empty(25) # empty lists to store corrected standard deviation per position
    lsr_crctd_data_mean=np.empty(25) # empty lists to store corrected mean per position
    lsr_crctd_data_mean_si=np.empty(25)
    lsr_crctd_data_stdev_si=np.empty(25)
    for mpos in range(0,25):
        new_data=[]
        new_data_sipm=[]
        ndata=0
        sipm_mean=np.mean(f2_nocut_si_filtered[mpos])
        sipm_stddev=np.std(f2_nocut_si_filtered[mpos])
        pmt_mean=np.mean(np.transpose(pmt_data)[mpos])
        pmt_stddev=np.std(np.transpose(pmt_data)[mpos])

        for run in range(len(pmt_data)):
            pmt_val=pmt_data[run][mpos]
            sipm_val=sipm_data[run][mpos]
            if pmt_val<pmt_mean+1*pmt_stddev and pmt_val>pmt_mean-1*pmt_stddev and sipm_val<sipm_mean+1*sipm_stddev and sipm_val>sipm_mean-1*sipm_stddev:
                new_data.append(pmt_val)
                new_data_sipm.append(sipm_val)

        lsr_crctd_data_stdev[mpos]=np.std(new_data)*10**6
        lsr_crctd_data_mean[mpos]=-np.mean(new_data)*10**6
        lsr_crctd_data_mean_si[mpos]=np.mean(new_data_sipm)*10**6
        lsr_crctd_data_stdev_si[mpos]=np.std(new_data_sipm)*10**6
        
    fig = plt.figure(figsize=(16, 10), dpi=80)
    # Error propagation formula for the ratio
    #err = (lsr_crctd_data_mean/pmt_mean_old) * np.sqrt((lsr_crctd_data_stdev / lsr_crctd_data_mean)**2 + (pmt_std_old / pmt_mean_old)**2)
    err = (lsr_crctd_data_stdev/pmt_mean_old)
    plt.errorbar(pos,lsr_crctd_data_mean/pmt_mean_old, yerr=lsr_crctd_data_stdev, fmt="o", color="k",capsize=5,label='Ratio of corrected to raw data')
    #plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber')
    plt.legend()
    plt.show()
    return lsr_crctd_data_mean,lsr_crctd_data_stdev,lsr_crctd_data_mean_si,lsr_crctd_data_stdev_si

def unpack_and_throw_outliers(filename,runs):
    pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data=np.transpose(np.array(pd.read_csv(filename)))[2:-(runs+4)]*10**6 # all data taken by the pmt per position
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-(runs+4):-4]*10**6 # all data taken by the sipm per position
    
    pmt_data_no_outliers=[]
    sipm_data_no_outliers=[]
    pmt_data_no_outliers_means=[]
    pmt_data_no_outliers_std=[]
    for i in range(25):
        data=np.transpose(pmt_data)[i]
        det_data=np.transpose(sipm_data)[i]
        # Create a box plot
        box_props = plt.boxplot(data)
        
       # plt.show()

        # Get the whisker ranges
        lower_whisker = box_props['whiskers'][0].get_ydata()[1]
        upper_whisker = box_props['whiskers'][1].get_ydata()[1]

        # Identify outliers using IQR method
        iqr = upper_whisker - lower_whisker
        lower_bound = lower_whisker
        upper_bound = upper_whisker

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        #print(outliers,lower_bound,upper_bound)
    
         # Remove outliers from the data
        no_outlier_pmt_data = -np.delete(data, outliers)
        no_outlier_sipm_data = np.delete(det_data, outliers)
        
        pmt_data_no_outliers.append(no_outlier_pmt_data)
        sipm_data_no_outliers.append(no_outlier_sipm_data)

        pmt_data_no_outliers_means.append(np.mean(no_outlier_pmt_data))
        pmt_data_no_outliers_std.append(np.std(no_outlier_pmt_data))
    
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    # Error propagation formula for the ratio
    #err = (lsr_crctd_data_mean/pmt_mean_old) * np.sqrt((lsr_crctd_data_stdev / lsr_crctd_data_mean)**2 + (pmt_std_old / pmt_mean_old)**2)
    err = (pmt_data_no_outliers_std/pmt_mean_old)
    plt.errorbar(pos,pmt_data_no_outliers_means/pmt_mean_old,yerr=err, fmt="o", color="k",capsize=5,label='Ratio of corrected to raw data')
    #plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber')
    plt.legend()
    plt.show()
    return         pmt_data_no_outliers, sipm_data_no_outliers, pmt_data_no_outliers_means, pmt_data_no_outliers_std
def unpack_and_throw_outliers_50(filename,runs):
    #pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    #pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data_raw=np.transpose(np.array(pd.read_csv(filename)))[2:-(runs+4)]*10**6 # all data taken by the pmt per position
    sipm_data_raw=np.transpose(np.array(pd.read_csv(filename)))[-(runs+4):-4]*10**6 # all data taken by the sipm per position
    
    pmt_data=[]
    sipm_data=[]
    pmt_mean_old=np.empty(25)
    pmt_std_old=np.empty(25)
    sipm_means=np.empty(25)
    sipm_std=np.empty(25)
    for i in range(25):
      #np.mean(np.concatenate(factory_pmt_data[0:25][i],factory_pmt_data[25:50][int(24-i)]))
      pmt_data.append(np.concatenate([pmt_data_raw[0:25][i],pmt_data_raw[25:50][24-i]]))
      pmt_mean_old[i]=np.mean(np.concatenate([pmt_data_raw[0:25][i],pmt_data_raw[25:50][24-i]]))
      pmt_std_old[i]=np.std(np.concatenate([pmt_data_raw[0:25][i],pmt_data_raw[25:50][24-i]]))

      sipm_data.append(np.concatenate([sipm_data_raw[0:25][i],sipm_data_raw[25:50][24-i]]))
      sipm_means[i]=np.mean(np.concatenate([sipm_data_raw[0:25][i],sipm_data_raw[25:50][24-i]]))
      sipm_std[i]=np.std(np.concatenate([sipm_data_raw[0:25][i],sipm_data_raw[25:50][24-i]]))
      print(np.std(np.concatenate([sipm_data_raw[0:25][i],sipm_data_raw[25:50][24-i]])))
    
    pmt_data=np.array(pmt_data)
    sipm_data=np.array(sipm_data)
    
    pmt_data_no_outliers=[]
    sipm_data_no_outliers=[]
    pmt_data_no_outliers_means=[]
    pmt_data_no_outliers_std=[]
    for i in range(25):
        data=np.transpose(pmt_data)[i]
        det_data=np.transpose(sipm_data)[i]
        # Create a box plot
        box_props = plt.boxplot(data)
        
       # plt.show()

        # Get the whisker ranges
        lower_whisker = box_props['whiskers'][0].get_ydata()[1]
        upper_whisker = box_props['whiskers'][1].get_ydata()[1]

        # Identify outliers using IQR method
        iqr = upper_whisker - lower_whisker
        lower_bound = lower_whisker
        upper_bound = upper_whisker

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        #print(outliers,lower_bound,upper_bound)
    
         # Remove outliers from the data
        no_outlier_pmt_data = -np.delete(data, outliers)
        no_outlier_sipm_data = np.delete(det_data, outliers)
        
        pmt_data_no_outliers.append(no_outlier_pmt_data)
        sipm_data_no_outliers.append(no_outlier_sipm_data)

        pmt_data_no_outliers_means.append(np.mean(no_outlier_pmt_data))
        pmt_data_no_outliers_std.append(np.std(no_outlier_pmt_data))
    
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    # Error propagation formula for the ratio
    #err = (lsr_crctd_data_mean/pmt_mean_old) * np.sqrt((lsr_crctd_data_stdev / lsr_crctd_data_mean)**2 + (pmt_std_old / pmt_mean_old)**2)
    err = (pmt_data_no_outliers_std)#/pmt_mean_old)
    plt.errorbar(pos,pmt_data_no_outliers_means,yerr=err, fmt="o", color="k",capsize=5,label='Ratio of corrected to raw data')
    #plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber')
    plt.legend()
    plt.show()
    return         pmt_data_no_outliers, sipm_data_no_outliers, pmt_data_no_outliers_means, pmt_data_no_outliers_std
def unpack_and_throw_outliers_other(filename,runs):
    pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data=np.transpose(np.array(pd.read_csv(filename)))[2:-(runs+4)]*10**6 # all data taken by the pmt per position
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-(runs+4):-4]*10**6 # all data taken by the sipm per position
    
    pmt_data_no_outliers=[]
    sipm_data_no_outliers=[]
    pmt_data_no_outliers_means=[]
    pmt_data_no_outliers_std=[]
    for i in range(25):
        data=np.transpose(pmt_data)[i]
        det_data=np.transpose(sipm_data)[i]
        # Create a box plot
        box_props = plt.boxplot(data)
        
       # plt.show()

        # Get the whisker ranges
        lower_whisker = box_props['whiskers'][0].get_ydata()[1]
        upper_whisker = box_props['whiskers'][1].get_ydata()[1]

        # Identify outliers using IQR method
        iqr = upper_whisker - lower_whisker
        lower_bound = lower_whisker
        upper_bound = upper_whisker

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        #print(outliers,lower_bound,upper_bound)
    
         # Remove outliers from the data
        no_outlier_pmt_data = -np.delete(data, outliers)
        no_outlier_sipm_data = np.delete(det_data, outliers)
        
        pmt_data_no_outliers.append(no_outlier_pmt_data)
        sipm_data_no_outliers.append(no_outlier_sipm_data)

        pmt_data_no_outliers_means.append(np.mean(no_outlier_pmt_data))
        pmt_data_no_outliers_std.append(np.std(no_outlier_pmt_data))
    
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    # Error propagation formula for the ratio
    #err = (lsr_crctd_data_mean/pmt_mean_old) * np.sqrt((lsr_crctd_data_stdev / lsr_crctd_data_mean)**2 + (pmt_std_old / pmt_mean_old)**2)
    err = (pmt_data_no_outliers_std/pmt_mean_old)
    plt.errorbar(pos,pmt_data_no_outliers_means/pmt_mean_old,yerr=err, fmt="o", color="k",capsize=5,label='Ratio of corrected to raw data')
    #plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber')
    plt.legend()
    plt.show()
    return         pmt_data_no_outliers, sipm_data_no_outliers, pmt_data_no_outliers_means, pmt_data_no_outliers_std

def unpack_and_throw_outliers_1_std(filename,runs):
    pmt_mean_old=-np.transpose(np.array(pd.read_csv(filename)))[-4]*10**6
    pmt_std_old=np.transpose(np.array(pd.read_csv(filename)))[-3]*10**6
    pmt_data=-np.transpose(np.array(pd.read_csv(filename)))[2:-(runs+4)]*10**6 # all data taken by the pmt per position
    sipm_data=np.transpose(np.array(pd.read_csv(filename)))[-(runs+4):-4]*10**6 # all data taken by the sipm per position
    
    pmt_data_no_outliers=[]
    sipm_data_no_outliers=[]
    pmt_data_no_outliers_means=[]
    pmt_data_no_outliers_std=[]
    for i in range(25):
        data=np.transpose(pmt_data)[i]
        det_data=np.transpose(sipm_data)[i]
        pmt_mean=pmt_mean_old[i]
        pmt_std=pmt_std_old[i]
        
        # Create a box plot
        box_props = plt.boxplot(data)
        
       # plt.show()

        # Get the whisker ranges
        lower_whisker = pmt_mean-pmt_std #box_props['whiskers'][0].get_ydata()[1]
        upper_whisker = pmt_mean+pmt_std #box_props['whiskers'][1].get_ydata()[1]

        # Identify outliers using IQR method
        iqr = upper_whisker - lower_whisker
        lower_bound = lower_whisker
        upper_bound = upper_whisker

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        #print(outliers,lower_bound,upper_bound)
    
         # Remove outliers from the data
        no_outlier_pmt_data = np.delete(data, outliers)
        no_outlier_sipm_data = np.delete(det_data, outliers)
        
        pmt_data_no_outliers.append(no_outlier_pmt_data)
        sipm_data_no_outliers.append(no_outlier_sipm_data)

        pmt_data_no_outliers_means.append(np.mean(no_outlier_pmt_data))
        pmt_data_no_outliers_std.append(np.std(no_outlier_pmt_data))
    
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    # Error propagation formula for the ratio
    #err = (lsr_crctd_data_mean/pmt_mean_old) * np.sqrt((lsr_crctd_data_stdev / lsr_crctd_data_mean)**2 + (pmt_std_old / pmt_mean_old)**2)
    err = (pmt_data_no_outliers_std/pmt_mean_old)
    plt.errorbar(pos,pmt_data_no_outliers_means/pmt_mean_old,yerr=err, fmt="o", color="k",capsize=5,label='Ratio of corrected to raw data')
    #plt.errorbar(pos,pmt_mean_old, yerr=pmt_std_old, fmt="o", color="b",capsize=5,label='Fiber')
    plt.legend()
    plt.show()
    return         pmt_data_no_outliers, sipm_data_no_outliers, pmt_data_no_outliers_means, pmt_data_no_outliers_std



# fitting functions D(x), R(x), and Chi2:
def P(val):
    return np.exp(val)/(1+np.exp(val))

def D(x,x0,prams):
    A,B,s1,s2=prams
    return A*np.exp(-s1*(x+x0))+B*np.exp(-s2*(x+x0))

def R(x,const,prams):
    x0,L=const
    A,B,s1,s2,pval=prams
    def Prob(val):
        return np.exp(val)/(1+np.exp(val))
    return Prob(pval)*(A*np.exp(-s1*(2*L-(x+x0)))+B*np.exp(-s2*(2*L-(x+x0))))

def chi2_fiber1(args):
    A,B,s1,s2=args
    return np.sum(((sumofexpdecays(pos,A,B,s1,s2)-fiber1_pmtavg_no_outlier)/fiber1_pmtstd_no_outlier)**2)
