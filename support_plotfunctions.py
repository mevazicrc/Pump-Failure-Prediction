# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:17:04 2018

@author: mevaz
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patches
from math import ceil, floor,sqrt
import os
import numpy as np
import pandas as pd
cwd = os.getcwd()


lstyle=[]

for lmode in ['-','--','-.']:
    for sym in 'o*^':
        for clr in ['b','r','g','c','m','k']:
            lstyle.append(clr+sym+lmode)




def plot_swin_segmented_well(api,segments_info,seg_data, data_swin, col_ts,col_swin,
                             date_lim, color_dict, prime_fail_short, plot_ts, normalize_ts, show_segs):

    api_seg_ids = segments_info[segments_info['api']==api].index.tolist()

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ymax = []
    ymin = []
    for j,key in enumerate(api_seg_ids):
        dataw = seg_data[key]
        k=0
        if plot_ts:
            for col in col_ts:
                k+=1
                y = dataw[col]
                if normalize_ts:
                    y = y/dataw[col].max()
                ymax.append(y.max())
                ymin.append(y.min())
                plt.plot_date(dataw.index,y,lstyle[k],label=col)

        for col in col_swin:
            k += 1
            y = data_swin[key][col]
            if col[-4:]=='unch':
                y = y/100
            ymax.append(y.max())
            ymin.append(y.min())
            plt.plot_date(data_swin[key].index,y,lstyle[k],label=col)

        if j==0:
            plt.legend(loc="upper right")
            plt.title(api)

    # DRAW RECTANGLE
    ymax = max(ymax)
    ymin = min(ymin)

    ylim = [ymin - 0.2 * (ymax - ymin), ymax+0.3*(ymax - ymin)]
    plt.ylim(ylim)


    if show_segs:
        for key in api_seg_ids:
            t1 = segments_info['SEGMENT_START'].loc[key]
            t2 = segments_info['SEGMENT_END'].loc[key]

            prime_fail = segments_info['PRIMARY_FAILURE'].loc[key]
            width = t2-t1
            height = 0.1 * (ymax - ymin)
            rect_loc0 = (t1, ymin - 0.2*(ymax - ymin) )
            rect1 = patches.Rectangle(rect_loc0, width, height,
                                                 color=color_dict[prime_fail])
            ax.add_patch(rect1)

            plt.plot_date([t1,t2,t2,t1,t1],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]],color_dict[prime_fail])
            if width>0.1*(date_lim[1]-date_lim[0]):
                plt.text(segments_info['SEGMENT_START'].loc[key], ymin - 0.175*(ymax - ymin) , prime_fail_short[prime_fail], size=20,
                     color='w')

    # plt.ylim([3000,7000])
    plt.xticks( rotation='vertical')
    plt.xlim(date_lim)
    plt.grid()
    plt.show()


def plot_fail_counts_with_time(primeFaileMode, secondFaileMode, segments_info, date_range, interval, barplot):
    tvec = []
    tvec_label = []
    fvec = []
    t2 = date_range[0]

    while t2 < date_range[1]:
        t1 = t2
        t2 = t1 + interval

        tmid = t1 + interval / 2
        tvec.append(tmid)
        if interval == pd.Timedelta('90d'):
            quarter = 'Q%d' % (int(tmid.month / 3) + 1)
            tvec_label.append('%s-%s' % (tmid.year, quarter))
        else:
            tvec_label.append('%s-%s' % (tmid.year, tmid.month))

        df_fail = segments_info[(segments_info['SEGMENT_END'] >= t1)
                                & (segments_info['SEGMENT_END'] < t2)]

        if len(primeFaileMode) > 0:
            df_fail = df_fail[df_fail['PRIMARY_FAILURE'] == primeFaileMode].groupby(
                ['SECONDARY_FAILURE'])[['SEGMENT_END']].count().sort_values(by='SEGMENT_END', ascending=False)

        if len(secondFaileMode) == 0:
            fvec.append(df_fail['SEGMENT_END'].sum())
        else:
            val = 0
            for sf in secondFaileMode:
                if sf in df_fail.index:
                    val = val + df_fail.loc[sf].values[0]

            fvec.append(val)

    plt.figure(figsize=(15, 10))
    width = 0.8
    if barplot:
        xticks = np.arange(1, len(fvec) + 1)
        plt.bar(xticks, fvec, width=width, color='g')
        xlim = [0.5, len(fvec) - 0.5]
    else:
        plt.plot_date(tvec, fvec, 'ko-', linewidth=2)
        xlim = date_range
        xticks = tvec_label

    plt.xticks(xticks, tvec_label, rotation=90)

    plt.xlim(xlim)
    plt.ylabel('Failure Count', fontweight='bold')
    plt.grid()

    ftitle = primeFaileMode
    if len(secondFaileMode) > 0:
        ftitle = ftitle + ' ['
        for sf in secondFaileMode:
            ftitle = ftitle + sf + ' + '
        ftitle = ftitle[:-3] + ']'

    plt.title(ftitle)
    plt.show()


def plot_annual_fail_counts(fail_annSt,RMT,fail_cols,trange, y_in_Percent):

    mainFailStat = fail_annSt[RMT][fail_cols]
    if y_in_Percent:
        mainFailStat = 100 * mainFailStat.div(mainFailStat.sum(axis=1), axis='rows')

    plt.figure(figsize=(15,10))
    width=0.5
    p=list()
    for i in range(len(fail_cols)):
        bottom=0
        if i>0:
            for j in range(i):
                bottom = bottom + mainFailStat[fail_cols[j]]
        p0 = plt.bar(mainFailStat.index, mainFailStat[fail_cols[i]], bottom=bottom,width=width)
        p.append(p0)

    xticks = np.arange(int(trange[0]),trange[1],1)
    plt.ylabel('Failure Count',fontweight='bold')
    if y_in_Percent:
        plt.ylabel('Failure Distribution (%)', fontweight='bold')

    plt.title('Failure summary (%s Active Wells)'%RMT,fontweight='bold')

    legend_list = [strc.replace("[ALL]", "") for strc in fail_cols]

    plt.legend(tuple(p), tuple(legend_list),loc="lower center")
    plt.xticks(xticks,rotation='vertical')
    plt.xlim(trange[0],trange[1])
    plt.show()

def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.0f}%".format(absolute)

def make_autopct(values, sumv):
    def my_autopct(pct):
        return "{:.0f}%".format(int(pct/100.*sumv))
    return my_autopct


def plot_failure_pies(FaileMode, segments_info, date_range, top_n, angle):
    df_fail = segments_info[(segments_info['SEGMENT_END'] >= date_range[0])
                            & (segments_info['SEGMENT_END'] < date_range[1])]

    df_fail = df_fail[df_fail['PRIMARY_FAILURE'] == FaileMode].groupby(
        ['SECONDARY_FAILURE'])[['SEGMENT_END']].count().sort_values(by='SEGMENT_END', ascending=False)

    newdf = pd.DataFrame({'count': df_fail['SEGMENT_END'].values.tolist()},
                         index=df_fail.index.values.tolist())

    nfails = newdf['count'].sum()

    # others
    new_row = pd.DataFrame(data={'count': [newdf['count'][top_n:].sum()]},
                           index=['OTHERS'])

    df_failtop = newdf.iloc[0:top_n].copy()
    data_top = pd.concat([df_failtop, new_row])

    data = data_top['count'].values.tolist()

    wedge_names = data_top.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))
    # wedges, texts, autotexts = ax.pie(data_topn,  wedgeprops=dict(width=1), startangle=angle, shadow=True,
    #                                   autopct='%1.0f%%')
    wedges, texts, autotexts = ax.pie(data, wedgeprops=dict(width=1), startangle=angle, shadow=True,
                                      autopct='%1.0f%%')

    # lambda pct: func(pct, data_all)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(wedge_names[i], xy=(x, y), xytext=(1.1 * np.sign(x), 1.2 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ptitle = "%d Events [%s] [%s-%s to %s-%s]" % (nfails, FaileMode,date_range[0].year,date_range[0].month,
                                                  date_range[1].year,date_range[1].month)
    plt.title(ptitle)

    plt.show()

def plot_cardsWithAnalog(wcards, wdata, col_dict, date_lims, ydims, color_dict, plt_dy, time2fail, t2fcolor_dict, divMax, ylog, ptitle, cardOnlyPreFailure):

    if type(wcards)==pd.DataFrame:
        fig=plt.figure(figsize=(10,20))
        plt.subplot(3,1,1)

        if cardOnlyPreFailure==True:
            trange = np.where(wcards[time2fail]>0)[0]
        else:
            trange = range(wcards.shape[0])

        nt = len(trange)
        for j in range(nt):
            t=trange[j]
            try:
                dhCard = eval(wcards['DownholeCardB'].iloc[t])
            except:
                continue
            load = [float(i) for i in dhCard[0]]
            pos =  [float(i) for i in dhCard[1]]

            plt.plot(pos,load,color=plt.cm.jet(j/nt))


    npl = len(col_dict)
    fig = plt.figure(figsize=(20, plt_dy*npl))
    ipl=0

    if 'Date' in wdata.columns:
        date_vec = wdata['Date']
    else:
        date_vec = wdata.index

    for i in list(col_dict.keys()):
        ipl+=1
        ax1 = plt.subplot(npl,1,ipl)


        for k in range(len(col_dict[i])):
            y = wdata[col_dict[i][k]]

            if divMax==True:
                y = y / y.max()

            if k == 0:
                minmaxy = [y.min(), y.max()]
            else:
                minmaxy[0] = min([minmaxy[0], y.min()])
                minmaxy[1] = max([minmaxy[1], y.max()])

            pdate = date_vec[np.where(~pd.isna(y))[0]]
            y = y.iloc[np.where(~pd.isna(y))[0]]

            if len(y)>500:
                d = int(len(y)/500)+1
                pdate = pdate[::d]
                y = y[::d]

            ax1.plot_date(pdate, y, color=color_dict[i][k], linewidth=3)

            if ylog==True:
                ax1.set_yscale('log')

            plt.xlim(date_lims)

        if len(col_dict[i])==1:
            plt.ylabel(col_dict[i][k])
        else:
            plt.legend(col_dict[i])

        if i<len(col_dict):
            plt.setp(ax1.get_xticklabels(), visible=False)
        if (ydims[i][1]-ydims[i][0])==0:
            ydims[i] = minmaxy
        plt.ylim(ydims[i])
        plt.grid(color='k')

        if len(time2fail)>0:
            ax2 = ax1.twinx()
            for k in range(len(time2fail)):
                ax2.plot_date(date_vec, wdata[time2fail[k]], '-',linewidth=5, color= t2fcolor_dict[k])
            ax2.set(ylim=(0.99, 2))
            plt.setp(ax2.get_yticklabels(), visible=False)
            ax2.grid(False)

            if i==0:
                ax2.legend(time2fail)

    mloc = mdates.MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = mdates.DateFormatter("%b '%y")
    ax1.xaxis.set_major_locator(mloc)
    ax1.xaxis.set_major_formatter(monthsFmt)
    ax1.grid(which='both',axis='both')
    plt.title(ptitle)
    fig.autofmt_xdate()
    plt.show()

def plot_clustered_card(card_data,col_x,col_y,data_labels,clust_name):
    num_clust = len(clust_name)
    ncolp = np.min([2, ceil(sqrt(num_clust))])
    nrowp = np.ceil(num_clust / ncolp)
    fig2 = plt.figure(figsize=(min([ncolp*6,30]), nrowp*5))

    load_labels = np.zeros((data_labels.shape[0], 1))
    inds_data = {}
    for clstr in range(num_clust):
        inds_clstr = np.where(data_labels == clstr)
        inds_clstr = inds_clstr[0]

        load_labels[inds_clstr] = clstr

        inds_data.update({clstr: inds_clstr})

        x = card_data[col_x][inds_clstr, :]
        y = card_data[col_y][inds_clstr, :]

        ax1 = plt.subplot(nrowp, ncolp, clstr + 1)
        for j in range(x.shape[0]):
            ax1.plot(x[j, :], y[j, :])
            ax1.tick_params(colors='w', direction='out')

        ax1.set_title(clust_name[clstr], color='w')
    plt.show()


def plot_cdf(Y,xlabel,ylabel,ylabel2):
    
    n_pdf = 100 * Y / np.sum(Y)
    
    x = 100 * np.linspace(0,1,len(Y))
    
    print(len(x))
    print(len(Y))



    fig1,ax1=plt.subplots()
    ax1.plot(x,np.cumsum(n_pdf),'k-',linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([0,100,0,100])
    plt.grid()
    plt.show()
    
    ax2 = ax1.twinx()
    plt.plot(x, Y,'r.') 
    ax2.set_ylabel(ylabel2,fontsize=12,color='r')
    
    plt.title(['Number of rod failures=' + str(np.sum(Y))])    
    
def plot_cardData(timeC,dataCard,Time, time2fail, time2fail_str, API):
    
    dataCard = dataCard[['Fillage','StrokeLength','Runtime','LoadLimit','PositionLimit' ,'Saved']]
    
    dataCard.describe()
    
    ncol = len(dataCard.columns)
    
    TLIM = [min(Time), max(Time)]    

    plt.rcParams.update({'font.size': 12})
    ppf = min([4,ncol]) # plot per figure

    nrowp = ceil(sqrt(ppf))
    ncolp  = int(ppf/nrowp)   
    
    print(nrowp)
    print(ncolp)
    for i in range(ncol):    
        colName = dataCard.columns[i]
        
        ip = (i+1)%ppf
        if ip==0:
            ip=ppf        
        if ip==1:
            fig=plt.figure(figsize=(15,10))
                
        ax1 = plt.subplot(nrowp,ncolp,ip)
        ax1.plot(timeC, dataCard[colName],'b.')
        plt.setp(ax1.get_xticklabels(), fontsize=12)
        ax1.set_ylabel(colName,fontsize=12,color='b')
        ax1.set( xlim = (TLIM[0],TLIM[1]), ylim = (0.99*min(dataCard[colName])-0.001,1.01*max(dataCard[colName])+0.001) )
        ax1.tick_params('y', colors='b')
        if ip==1:
            plt.title(API)
            
        ax2 = ax1.twinx()
        ax2.plot(Time, time2fail,'r.') 
        if ip==1:
            ax2.set_ylabel(time2fail_str,fontsize=12,color='r')
            ax2.set_xlabel('Time',fontsize=12)
            
        ax2.tick_params('y', colors='r',labelright='off')
        ax2.set( ylim = (0.9,1.1) )    

        fig.tight_layout()
        plt.grid()
        plt.show()
    fname = 'figs/' + str(API) +'.pdf'
    if os.path.isfile(fname):
        os.remove(fname)
    fig.savefig(fname)

    
def plot_analogData( Time, dataAnalog, time2fail, time2fail_str, API):
    
    dataAnalog = dataAnalog[['CurTbgPSI','CurAGAGas','YestAGAGas','CurDP','CurCsgPSI',
                             'CurRPM','StrokeCnt','CurMaxLB','CurMinLB',
                             'CurIP','CurCycles','CurPercRT','CurFillage',
                              'CurProd','CurOrficeDiameter']]
 
    

    ncol = len(dataAnalog.columns)    
    fail_time = Time[time2fail==1]
    
    TLIM = [max([min(Time),min(fail_time)-0.5]), min([max(Time),max(fail_time)+0.5])]
    TLIM = [min(Time), max(Time)]    


    plt.rcParams.update({'font.size': 12})
    ppf = min([9,ncol]) # plot per figure

    nrowp = ceil(sqrt(ppf))
    ncolp  = int(ppf/nrowp)   
    
    print(nrowp)
    print(ncolp)
    for i in range(ncol):    
        colName = dataAnalog.columns[i]
        
        ip=(i+1)%ppf
        if ip==0:
            ip=ppf        
        if ip==1:
            fig=plt.figure(figsize=(15,10))
        
        ax1 = plt.subplot(nrowp,ncolp,ip)
        ax1.plot(Time, dataAnalog[colName],'b.')
        plt.setp(ax1.get_xticklabels(), fontsize=12)
        ax1.set_ylabel(colName,fontsize=12,color='b')
        if (not np.isnan(0.99*min(dataAnalog[colName])) ) and (not np.isinf(0.99*min(dataAnalog[colName]))):
                ax1.set( xlim = (TLIM[0],TLIM[1]), ylim = (0.99*min(dataAnalog[colName])-0.001,1.01*max(dataAnalog[colName])+0.001) )
                
        ax1.tick_params('y', colors='b')
        if ip==1:
            plt.title(API)
            
        ax2 = ax1.twinx()
        ax2.plot(Time, time2fail,'r.') 
        if ip==1:
            ax2.set_ylabel(time2fail_str,fontsize=12,color='r')
            ax2.set_xlabel('Time',fontsize=12)
            
        ax2.tick_params('y', colors='r')
        ax2.set( ylim = (0.9,1.1) )    

        fig.tight_layout()
        plt.grid()
        plt.show()
    fname = 'figs/' + str(API) +'.pdf'
    if os.path.isfile(fname):
        os.remove(fname)
    fig.savefig(fname)
    
def plot_Data( Time, U, time2fail, time2fail_str, API):
    
    ncol = len(U.columns)    
    fail_time = Time[time2fail==1]
    
    TLIM = [max([min(Time),min(fail_time)-0.5]), min([max(Time),max(fail_time)+0.5])]
    TLIM = [min(Time), max(Time)]    


    plt.rcParams.update({'font.size': 12})
    ppf = min([9,ncol]) # plot per figure

    nrowp = ceil(sqrt(ppf))
    ncolp  = int(ppf/nrowp)   
    
    print(nrowp)
    print(ncolp)
    for i in range(ncol):    
        colName = U.columns[i]
        
        ip=(i+1)%ppf
        if ip==0:
            ip=ppf        
        if ip==1:
            fig=plt.figure(figsize=(15,10))
        
        ax1 = plt.subplot(nrowp,ncolp,ip)
        ax1.plot(Time, U[colName],'b.')
        plt.setp(ax1.get_xticklabels(), fontsize=12)
        ax1.set_ylabel(colName,fontsize=12,color='b')
        if (not np.isnan(0.99*min(U[colName])) ) and (not np.isinf(0.99*min(U[colName]))):
                ax1.set( xlim = (TLIM[0],TLIM[1]), ylim = (0.99*min(U[colName])-0.001,1.01*max(U[colName])+0.001) )
                
        ax1.tick_params('y', colors='b')
        if ip==1:
            plt.title(API)
            
        ax2 = ax1.twinx()
        ax2.plot(Time, time2fail,'r.') 
        if ip==1:
            ax2.set_ylabel(time2fail_str,fontsize=12,color='r')
            ax2.set_xlabel('Time',fontsize=12)
            
        ax2.tick_params('y', colors='r')
        ax2.set( ylim = (0.9,1.1) )    

        fig.tight_layout()
        plt.grid()
        plt.show()
    fname = 'figs/' + str(API) +'.pdf'
    if os.path.isfile(fname):
        os.remove(fname)
    fig.savefig(fname)

def plot_chemData(Time, dataChem, time2fail, time2fail_str, API):

    ncol = len(dataChem.columns)
    
    fail_time = Time[time2fail==1]
    
    TLIM = [max([min(Time),min(fail_time)-0.5]), min([max(Time),max(fail_time)+0.5])]
    
#    %matplotlib qt


    plt.rcParams.update({'font.size': 12})
    ppf = min([9,ncol]) # plot per figure

    nrowp = ceil(sqrt(ppf))
    ncolp  = int(ppf/nrowp)   
    
    print(nrowp)
    print(ncolp)
    for i in range(ncol):    
        colName = dataAnalog.columns[i]
        
        ip=(i+1)%ppf
        if ip==0:
            ip=ppf        
        if ip==1:
            fig=plt.figure(figsize=(15,10))
        
        ax1 = plt.subplot(nrowp,ncolp,ip)
        ax1.plot(Time, dataAnalog[colName],'b.')
        plt.setp(ax1.get_xticklabels(), fontsize=12)
        ax1.set_ylabel(colName,fontsize=12,color='b')
        ax1.set( xlim = (TLIM[0],TLIM[1]) )
        ax1.tick_params('y', colors='b')
        
        ax2 = ax1.twinx()
        ax2.plot(Time, time2fail,'r.') 
        if ip==1:
            ax2.set_ylabel(time2fail_str,fontsize=12,color='r')
            ax2.set_xlabel('Time',fontsize=12)
            
        ax2.tick_params('y', colors='r')
        ax2.set( ylim = (0.9,1.1) )    

        fig.tight_layout()
        plt.grid()
        plt.show()
    fname = 'figs/Chemdata' + str(API) +'.pdf'
    if os.path.isfile(fname):
        os.remove(fname)
    fig.savefig(fname)
