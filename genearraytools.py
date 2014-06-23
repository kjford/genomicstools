'''
Copyright Kevin Ford (2014)
genearraytools.py

Tools for normalization, reduction, and heirarchical clustering of microarray data
Clustering plot is borrowed from:
Copyright 2005-2012 J. David Gladstone Institutes, San Francisco California
Author Nathan Salomonis - nsalomonis@gmail.com

'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import matplotlib as mpl

import string
import time
import re



def loadFromTxt(filename):
    # loads microarray data form tab delimited txt file
    # returns panda DataFrame
    print('Reading data from %s...'%filename)
    data=pd.read_table(filename)
    nrows,ncol=data.shape
    print('Data has %i rows and %i columns'%(nrows,ncol))
    return data

def reindexData(DF):
    # reindex the data frame by splitting runs by condition
    # makes use of multilevel indexing
    # assumptions:
    # first 2 columns are UNIQID and NAME
    conditions=list(x.split()[0] for x in DF.columns.tolist()[2:])
    runs=list(re.split('\(|\)',x)[1] for x in DF.columns.tolist()[2:])
    indexer=[DF.UNIQID,DF.NAME]
    leveler=[np.array(conditions),np.array(runs)]
    df2=pd.DataFrame(np.array(DF.iloc[:,2:]),index=indexer, columns=leveler)
    return df2
    
def inspectTechRep(DF,colname,doplot=True):
    # inspect technical replicates in a given column
    # plots each replicate (duplicated gene name) vs the mean of the replicate
    # note: want to do this before reindexing
    genevals=np.zeros(DF.shape[0])
    meanvals=np.zeros(DF.shape[0])
    genestd=[]
    genenames=[]
    c=0
    for k,i in DF.groupby('NAME'):
        genenames.append(k)
        vals=i[colname]
        meanvals[c:c+len(vals)]=np.mean(vals)
        genevals[c:c+len(vals)]=vals
        if len(vals)>0:
            genestd.append(np.std(vals))
        else:
            genestd.append(0)
        c+=len(vals)
    if doplot:
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(meanvals,genevals,'k.')
        ax.set_title(colname)
        ax.set_xlabel('Mean Gene Value')
        ax.set_ylabel('All Gene Values')
        fig.show()
    return (genevals,meanvals,genenames,genestd)
    
def inspectExpRep(DF,condname,savename=None):
    # inspect experimental replicates given condition name
    # plots each matrix of each replicate (duplicated gene name) vs others
    # note: want to do this after reindexing
    dfsub=DF[condname]
    colnames=dfsub.columns.tolist()
    ng,nrep=dfsub.shape
    cmat=np.zeros((nrep,nrep))
    fig=plt.figure()
    for i in range(nrep):
        vali=dfsub.ix[:,i]
        for j in range(nrep):
            if j<i:
                ax=fig.add_subplot(nrep,nrep,(nrep*i)+(j+1))
                valj=dfsub.ix[:,j]
                ax.plot(vali,valj,'k.')
                #ax.set_xlabel(colnames[i])
                #ax.set_ylabel(colnames[j])
                ax.set_xlabel(i)
                ax.set_ylabel(j)
                plt.setp(ax.get_yticklines(),visible=False)
                plt.setp(ax.get_xticklines(),visible=False)
                plt.setp(ax.get_yticklabels(),visible=False)
                plt.setp(ax.get_xticklabels(),visible=False)
                cmat[i,j]=np.correlate(np.array(vali),np.array(valj))
    if savename:
        fig.savefig(savename)
    fig.show()
    return cmat

def cleanArrayData(DF):
    # take median of technical replicates to get one value per gene
    # subtract median of each column to normalize
    # output is reindexed by gene name with 2 level column indices
    # note: want to do this before reindexing
    
    # take median of each technical replicate
    genenames=[]
    dataholder=[]
    for gene,vals in DF.groupby('NAME'):
        genenames.append(gene)
        dataholder.append(np.array(vals.median()))
    dataholder=np.array(dataholder)
    conditions=list(x.split()[0] for x in DF.columns.tolist()[2:])
    runs=list(re.split('\(|\)',x)[1] for x in DF.columns.tolist()[2:])
    leveler=[np.array(conditions),np.array(runs)]
    dfout=pd.DataFrame(dataholder,index=genenames,columns=leveler)
    dfout=dfout-dfout.median()
    return dfout
        

def heatmap(DF,column_metric='euclidean',row_metric='cityblock',column_method='single',
            row_method='average',filename='clustered.pdf'):
    '''
    Hierarchical clustering by row and column
    plots heat map (red white blue)
    returns flattened clusters
    Taken from:
    Copyright 2005-2012 J. David Gladstone Institutes, San Francisco California
    Author Nathan Salomonis - nsalomonis@gmail.com
    '''
    n,m = DF.shape
    cmap=plt.cm.bwr
    x=np.array(DF.iloc[:,2:]) # get data assuming first 2 columns are names and ids
    vmin=x.min()
    vmax=x.max()
    vmax = max([vmax,abs(vmin)])
    vmin = vmax*-1
    norm = mpl.colors.Normalize(vmin/2, vmax/2) ### adjust the max and min to scale these colors
    default_window_hight = 8.5
    default_window_width = 12
    fig = plt.figure(figsize=(default_window_width,default_window_hight))
    color_bar_w = 0.015
    ## calculate positions for all elements
    # ax1, placement of dendrogram 1, on the left of the heatmap
    [ax1_x, ax1_y, ax1_w, ax1_h] = [0.05,0.22,0.2,0.6]   ### The second value controls the position of the matrix relative to the bottom of the view
    width_between_ax1_axr = 0.004
    height_between_ax1_axc = 0.004 ### distance between the top color bar axis and the matrix
    
    # axr, placement of row side colorbar
    [axr_x, axr_y, axr_w, axr_h] = [0.31,0.1,color_bar_w,0.6] ### second to last controls the width of the side color bar - 0.015 when showing
    axr_x = ax1_x + ax1_w + width_between_ax1_axr
    axr_y = ax1_y; axr_h = ax1_h
    width_between_axr_axm = 0.004

    # axc, placement of column side colorbar
    [axc_x, axc_y, axc_w, axc_h] = [0.4,0.63,0.5,color_bar_w] ### last one controls the hight of the top color bar - 0.015 when showing
    axc_x = axr_x + axr_w + width_between_axr_axm
    axc_y = ax1_y + ax1_h + height_between_ax1_axc
    height_between_axc_ax2 = 0.004

    # axm, placement of heatmap for the data matrix
    [axm_x, axm_y, axm_w, axm_h] = [0.4,0.9,2.5,0.5]
    axm_x = axr_x + axr_w + width_between_axr_axm
    axm_y = ax1_y; axm_h = ax1_h
    axm_w = axc_w

    # ax2, placement of dendrogram 2, on the top of the heatmap
    [ax2_x, ax2_y, ax2_w, ax2_h] = [0.3,0.72,0.6,0.15] ### last one controls hight of the dendrogram
    ax2_x = axr_x + axr_w + width_between_axr_axm
    ax2_y = ax1_y + ax1_h + height_between_ax1_axc + axc_h + height_between_axc_ax2
    ax2_w = axc_w

    # axcb - placement of the color legend
    [axcb_x, axcb_y, axcb_w, axcb_h] = [0.07,0.88,0.18,0.09]
    
    x=np.array(DF)

    # Compute and plot top dendrogram
    start_time = time.time()
    d2 = dist.pdist(x.T)
    D2 = dist.squareform(d2)
    ax2 = fig.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
    Y2 = sch.linkage(D2, method=column_method, metric=column_metric) ### array-clustering metric - 'average', 'single', 'centroid', 'complete'
    Z2 = sch.dendrogram(Y2)
    ind2 = sch.fcluster(Y2,0.7*max(Y2[:,2]),'distance') ### This is the default behavior of dendrogram
    ax2.set_xticks([]) ### Hides ticks
    ax2.set_yticks([])
    time_diff = str(round(time.time()-start_time,1))
    print('Column clustering completed in %s seconds' % time_diff)
    
    # Compute and plot left dendrogram.
    start_time = time.time()
    d1 = dist.pdist(x)
    D1 = dist.squareform(d1)  # full matrix
    ax1 = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True) # frame_on may be False
    Y1 = sch.linkage(D1, method=row_method, metric=row_metric) ### gene-clustering metric - 'average', 'single', 'centroid', 'complete'
    Z1 = sch.dendrogram(Y1, orientation='right')
    ind1 = sch.fcluster(Y1,0.7*max(Y1[:,2]),'distance') ### This is the default behavior of dendrogram
    ax1.set_xticks([]) ### Hides ticks
    ax1.set_yticks([])
    time_diff = str(round(time.time()-start_time,1))
    print('Row clustering completed in %s seconds' % time_diff)
    
    # Plot distance matrix.
    axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h])  # axes for the data matrix
    xt = x
    idx2 = Z2['leaves'] ### apply the clustering for the array-dendrograms to the actual matrix data
    xt = xt[:,idx2]
    ind2 = ind2[:,idx2] ### reorder the flat cluster to match the order of the leaves the dendrogram
    idx1 = Z1['leaves'] ### apply the clustering for the gene-dendrograms to the actual matrix data
    xt = xt[idx1,:]   # xt is transformed x
    ind1 = ind1[idx1,:] ### reorder the flat cluster to match the order of the leaves the dendrogram
    ### taken from http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python/3011894#3011894
    im = axm.matshow(xt, aspect='auto', origin='lower', cmap=cmap, norm=norm) ### norm=norm added to scale coloring of expression with zero = white or black
    axm.set_xticks([]) ### Hides x-ticks
    axm.set_yticks([])
    
    # Add text
    new_row_header=[]
    new_column_header=[]
    column_header=np.array(DF.columns.tolist())
    if column_header.ndim>1: # multilevel indexed, use first level
        column_header=column_header[:,0]
    row_header=np.array(DF.index.values)
    
    for i in range(n):
        if len(row_header)<100: ### Don't visualize gene associations when more than 100 rows
            if type(row_header[idx1[i]])==type((0,0)):
                row_header[idx1[i]] = row_header[idx1[i]][1]
            axm.text(x.shape[1]-0.5, i, '  '+row_header[idx1[i]])
        new_row_header.append(row_header[idx1[i]])
        
    for i in range(m):
        axm.text(i, -0.9, ' '+column_header[idx2[i]]+str(idx2[i]), rotation=270, verticalalignment="top") # rotation could also be degrees
        new_column_header.append(column_header[idx2[i]]+str(idx2[i]))
        
    # Plot colside colors
    # axc --> axes for column side colorbar
    axc = fig.add_axes([axc_x, axc_y, axc_w, axc_h])  # axes for column side colorbar
    cmap_c = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
    dc = np.array(ind2, dtype=int)
    dc.shape = (1,len(ind2)) 
    im_c = axc.matshow(dc, aspect='auto', origin='lower', cmap=cmap_c)
    axc.set_xticks([]) ### Hides ticks
    axc.set_yticks([])
    
    # Plot rowside colors
    # axr --> axes for row side colorbar
    axr = fig.add_axes([axr_x, axr_y, axr_w, axr_h])  # axes for column side colorbar
    dr = np.array(ind1, dtype=int)
    dr.shape = (len(ind1),1)
    #print ind1, len(ind1)
    cmap_r = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
    im_r = axr.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
    axr.set_xticks([]) ### Hides ticks
    axr.set_yticks([])

    # Plot color legend
    axcb = fig.add_axes([axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)  # axes for colorbar
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='horizontal')
    axcb.set_title("colorkey")
    
    cb.set_label("Differential Expression (log2 fold)")
    exportFlatClusterData(filename, new_row_header,new_column_header,xt,ind1,ind2)

    ### Render the graphic
    if len(row_header)>30 or len(column_header)>30:
        plt.rcParams['font.size'] = 5
    else:
        plt.rcParams['font.size'] = 8

    plt.savefig(filename)
    print('Exporting: %s'%filename)
    
    plt.show()
    # return back to default plot parameters
    plt.rcdefaults()


def exportFlatClusterData(filename, new_row_header,new_column_header,xt,ind1,ind2):
    """ 
    Export the clustered results as a text file, only indicating the flat-clusters rather than the tree
    Taken from:
    Copyright 2005-2012 J. David Gladstone Institutes, San Francisco California
    Author Nathan Salomonis - nsalomonis@gmail.com
    """
    
    filename = string.replace(filename,'.pdf','.txt')
    export_text = open(filename,'w')
    column_header = string.join(['UID','row_clusters-flat']+new_column_header,'\t')+'\n' ### format column-names for export
    export_text.write(column_header)
    column_clusters = string.join(['column_clusters-flat','']+ map(str, ind2),'\t')+'\n' ### format column-flat-clusters for export
    export_text.write(column_clusters)
    
    ### The clusters, dendrogram and flat clusters are drawn bottom-up, so we need to reverse the order to match
    new_row_header = new_row_header[::-1]
    xt = xt[::-1]
    
    ### Export each row in the clustered data matrix xt
    i=0
    for row in xt:
        if type(new_row_header[i])==type((0,0)):
            new_row_header[i]=new_row_header[i][1]
        export_text.write(string.join([new_row_header[i],str(ind1[i])]+map(str, row),'\t')+'\n')
        i+=1
    export_text.close()
    
    ### Export as CDT file
    filename = string.replace(filename,'.txt','.cdt')
    export_cdt = open(filename,'w')
    column_header = string.join(['UNIQID','NAME','GWEIGHT']+new_column_header,'\t')+'\n' ### format column-names for export
    export_cdt.write(column_header)
    eweight = string.join(['EWEIGHT','','']+ ['1']*len(new_column_header),'\t')+'\n' ### format column-flat-clusters for export
    export_cdt.write(eweight)
    
    ### Export each row in the clustered data matrix xt
    i=0
    for row in xt:
        export_cdt.write(string.join([new_row_header[i]]*2+['1']+map(str, row),'\t')+'\n')
        i+=1
    export_cdt.close()