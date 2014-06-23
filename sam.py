'''
sam.py
Copyright Kevin Ford (2014)

Statistical Analysis of Microarray Data
as per http://statweb.stanford.edu/~tibs/SAM/

works with pandas DataFrames
with experimental replicates and multiclass

includes overrepresentation analysis class
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpmath import binomial as nCk
from collections import Counter

class SAM:
    '''
    SAM object
    initialize with dataframe containing columns with level0 name condition and level1 rep name
    '''
    def __init__(self, DF, alpha=99.0, delta=0.3, nperm=500, randseed=31415, s0=-1, s0per=-1):
        self.data= np.array(DF) # dataframe
        self.alpha=alpha
        self.delta=delta
        self.nperm=nperm
        self.randseed=randseed
        self.s0per=s0per
        self.s0=s0
        # get sizes
        r,c=self.data.shape
        self.n_i= r #num genes
        self.n_samp = c
        self.n_g= len(DF.columns.levels[0]) #num conditions
        condnames = [x[0] for x in DF.columns.tolist()]
        self.condnames=condnames
        # index samples by condition
        groups,grind=np.unique(condnames,return_inverse=True)
        self.groups = groups # sorted
        self.grind = grind
        nk = np.array([condnames.count(foo) for foo in groups])
        self.n_k= nk #num samples in each condition
        
        self.genes= np.array(DF.index.values)
        
    def __repr__(self):
        return 'Statistical Analysis of Microarray Data object'
    
    def get_rs(self,xi,ck):
        # depreciated: use vectorized version
        # xi is values for genei at each condition
        # indexed by condition in ck
        # average over all:
        xbari= np.mean(xi)
        # average in each condition:
        xbarik = np.array([np.mean(xi[ck==foo]) for foo in np.unique(ck)])
        # n for each condition
        nk=self.n_k
        # compute ri across group variance score
        rinorm=nk.sum()/(1.0*nk.prod())
        ri=(rinorm*(nk*(xbarik-xbari)**2).sum())**0.5
        # compute si within group variance
        xijss = np.array([np.sum((xi[ck==foo] - xbarik[foo])**2) for foo in np.unique(ck)])
        si=((1.0/((nk-1).sum())) * (1.0/(nk.sum())) * xijss.sum())**0.5
        return (ri,si)
    
    def get_rs_vec(self,x,ck):
        # x is values for each genei at each sample
        # indexed by condition in ck
        # n for each condition
        nk=self.n_k
        nk=nk.reshape(nk.size,1) # to allow for broadcasting
        # average over all:
        xbar=x.mean(axis=1)
        # average in each condition:
        xbark = np.array([x[:,(ck==foo)].mean(axis=1) for foo in np.unique(ck)]) # nk x i
        # compute ri across group stdev
        if self.n_g>2:
            rinorm=nk.sum()/(1.0*nk.prod()) # per paper
            # alternate for F-test
            # rinorm = 1.0/(self.n_g*1.0 - 1.0) 
            ri=(rinorm*(nk*(xbark-xbar)**2).sum(axis=0))**0.5
        else:
            ri=xbark[0]-xbark[1]
        # compute si within group stdev
        xijss = np.array([((x[:,(ck==foo)] - xbark[foo].reshape(xbark.shape[1],1))**2).sum(axis=1) for foo in np.unique(ck)])
        si=((1.0/((nk-1.0).sum())) * np.sum(1.0/nk) * xijss.sum(axis=0))**0.5
        return (ri,si)
    
    def get_s0(self,s,r):
        # computes the exchangeability factor
        # 3 options: define value, percentage or compute
        # based on minimization of CV of d
        if self.s0>=0: #already have one or given
            s0=self.s0
        elif self.s0per>=0: # define percentage
            s0=np.percentile(s,self.s0per)
            self.s0=s0
        else: # compute
            perrange=np.arange(100)
            s0range=np.zeros((100,len(r)))
            # compute d at each s0 as percentile of s in 1% increments
            for i in perrange:
                s0range[i,:]=r/(s+1.0*np.percentile(s,i))
            # minimize CV of median absolute deviation of d values over 4 width sliding window
            cvs=np.zeros(len(perrange[2:-2]))
            for i in perrange[2:-2]:
                dev=0.64*np.median(np.abs(s0range[(i-2):(i+2),:]-np.median(s0range[(i-2):(i+2),:],axis=0)),axis=1)
                cvs[i-2]=dev.std()/dev.mean()
            # get minimum
            minper=cvs.argmin() + 2
            s0=np.percentile(s,minper)
            self.s0=s0
            self.s0per=minper
        return s0
    
    def get_d(self, x, ck):
        # compute d for each gene in x with groupings ck
        r,s=self.get_rs_vec(x,ck)
        s0=self.get_s0(s,r)
        return (r/(s+s0))
        
    def run(self):
        # runs SAM on data
        # compute d for each gene
        x=self.data
        ck=self.grind
        print('Computing scores...')
        d=self.get_d(x,ck)
        self.dvals = d
        print('Done.')
        # sort (decending order)
        dsorted=np.sort(d)[::-1]
        self.dsorted=dsorted
        sortinds = np.argsort(-d)
        self.rankedinds = sortinds
        # compute d on random permutations of samples
        np.random.seed(self.randseed)
        db=np.zeros((self.nperm,self.n_i))
        print('Performing permutations to compute significance...')
        for i in range(self.nperm):
            cki=np.random.permutation(ck)
            di=self.get_d(x,cki)
            di.sort()
            db[i,:]=di[::-1]
        self.dbar= db.sum(axis=0)/(self.nperm*1.0)
        self.dperms=db
        print('Done.')
        # compute significance:
        self.computeSig()
    
    def computeSig(self):
        # if there is an alpha (significance level), set delta
        if self.alpha:
            qalpha=np.percentile(self.dperms[:,0],self.alpha)
            self.delta=qalpha-self.dbar[0]
        
        # number of genes above delta
        self.nsig=np.sum(np.abs(self.dsorted-self.dbar)>=self.delta)
        # gene names of significant genes
        self.siginds=self.rankedinds[np.abs(self.dsorted-self.dbar)>=self.delta]
        self.siggenes = self.genes[self.siginds]
        # estimated number of false positives
        FP= np.sum((np.abs(self.dperms-self.dbar)>=self.delta),axis=1)
        # take median and 90% quartile
        FP50 = np.median(FP)
        FP90 = np.percentile(FP,90.0)
        # compute pi0 (proportion of true unaffected genes):
        if self.n_g>2:
            qlow=np.min(self.dperms)
            qhigh=np.median(self.dperms)
        else:
            qlow=np.percentile(self.dperms,25.0)
            qhigh=np.percentile(self.dperms,75.0)
        pi0=2.0*np.sum(np.logical_and(self.dvals<=qhigh,self.dvals>=qlow))/(1.0*self.n_i)
        self.pi0=np.minimum(pi0,1)
        # false positive rate:
        self.FPR50 = (FP50*self.pi0)/(1.0*self.nsig)
        self.FPR90 = (FP90*self.pi0)/(1.0*self.nsig)
        print('Delta level: %2f'%self.delta)
        print('Hits: %i\n FPR(50): %2f\n FPR(90): %2f'%(self.nsig,self.FPR50,self.FPR90))
        # plot expected D vs computed with cutoff boundaries defined by delta
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(self.dbar,self.dsorted,'k.')
        ax.plot([self.dsorted[0],self.dsorted[-1]],[self.dsorted[0],self.dsorted[-1]],'r--')
        ax.plot([self.dsorted[0],self.dsorted[-1]],self.delta+np.array([self.dsorted[0],self.dsorted[-1]]),'r')
        ax.plot([self.dsorted[0],self.dsorted[-1]],-self.delta+np.array([self.dsorted[0],self.dsorted[-1]]),'r')
        ax.set_title('Scores')
        ax.set_ylabel('Actual D')
        ax.set_xlabel('Expected D')
        fig.show()


class ORA:
    '''
    Overrepresentation analysis (ORA)
    Initialize with a list of labels associated with gene names (e.g. GO terms)
    Assess significance (using hypergeometric p-value) of supplied gene list
    '''
    def __init__(self,labels,genes,plotgroups=None,useHG=False,nperm=10000,BFcorr=True):
        self.allterm=np.array(labels)
        self.allgenes=np.array(genes)
        self.parseGroups(labels)
        self.testset=[]
        self.testgroups={}
        self.L=len(labels)
        self.plotgroups=plotgroups
        self.useHG=useHG
        self.nperm=nperm
        self.BFcorr=BFcorr # display Bonferroni corrected p CIs on plot?
    
    def __repr__(self):
        return 'Overrepresentation analysis object'
        
    def parseGroups(self,labels):
        # group the term labels and counts in each
        c=Counter(labels)
        binned=c.most_common()
        self.terms=[x[0] for x in binned]
        self.termcounts=[x[1] for x in binned]
    
    def hgpval(self,nk,Ns,Na):
        # compute hypergeometric probability of seeing nk items in set s of size Ns
        # with a list of size Na
        iterover=np.minimum(self.L,Ns)+1
        p=0
        for i in np.arange(nk,iterover):
            p+=((nCk(Ns,i)*nCk(self.L-Na,Ns-i))/nCk(self.L,Ns))
        return p
    
    def permpval(self,nk,Ns,Na):
        # compute significance by performing permutation tests
        # more effective when groups sizes are large since hypergeometric loses precision
        # implemented using binomial random draws
        draws=np.random.binomial(Na,Ns/(1.0*self.L),self.nperm)
        # p is the number of draws>= observed nk genes in list
        p=np.sum(draws>=nk)/(1.0*len(draws))
        # return the 95% upper and lower bounds
        p5=np.percentile(draws,5.0)
        p95=np.percentile(draws,95.0)
        # return Bonferroni corrected 5 and 95 intervals
        p5_corr=np.percentile(draws,5.0/len(self.terms))
        p95_corr=np.percentile(draws,100.0-5.0/len(self.terms))
        return (p,(p5,p95),(p5_corr,p95_corr))
    
    def testGroup(self,termid):
        # compute significance for items in test set that are on set termid
        setgene=set(self.testset)
        term=self.terms[termid]
        setterm=set(self.allgenes[self.allterm==term])
        inlist=set.intersection(setgene,setterm)
        self.testgroups[term]=inlist
        if self.useHG:
            p=self.hgpval(len(inlist),self.termcounts[termid],len(self.testset))
            return p
        else:
            p,ci,ci_corr=self.permpval(len(inlist),self.termcounts[termid],len(self.testset))
            return p,ci,ci_corr
    
    def run(self,X):
        # run analysis over genes in test set X which is a list of gene names
        self.testset=X
        nterms=len(self.terms)
        ps=[]
        if self.useHG:
            for i in range(nterms):
                ps.append(self.testGroup(i))
        else:
            cis=[]
            ci_cs=[]
            for i in range(nterms):
                pi,ci,ci_c=self.testGroup(i)
                ps.append(pi)
                cis.append(ci)
                ci_cs.append(ci_c)
            self.CI = cis
            self.CI_corr=ci_cs
        self.pvals=ps
        self.pvals_corr=list(np.array(ps)*1.0*nterms)
        
        self.makeplots()
        
    def makeplots(self):
        # plot bar graphs of over representation in each group
        # note: want to keep this to a reasonable number of groups
        if self.plotgroups: # default to all if not specified
            nbars=self.plotgroups
        else:
            nbars=len(self.terms)
        # order by p value
        barorder=np.argsort(self.pvals)
        nk=[]
        nexpected=[]
        terms=[]
        ci_low=[]
        ci_hi=[]
        for i in range(nbars):
            ns=self.termcounts[barorder[i]]
            terms.append(self.terms[barorder[i]])
            nk.append(len(self.testgroups[terms[i]]))
            # get expected counts
            nexpected.append((ns/(1.0*self.L))*len(self.testset))
            if ~self.useHG:
                # add the CIs
                if self.BFcorr:
                    ci_low.append(self.CI_corr[barorder[i]][0])
                    ci_hi.append(self.CI_corr[barorder[i]][1])
                else:
                    ci_low.append(self.CI[barorder[i]][0])
                    ci_hi.append(self.CI[barorder[i]][1])
        # plot
        orval=np.array(nk)/np.array(nexpected)
        plt.figure()
        plt.barh(range(nbars),orval,align='center')
        plt.plot([1,1],[-1,nbars],'r--')
        if ~self.useHG:
            # plot CIs
            xlo=1.0-np.array(ci_low)/np.array(nexpected)
            xhi=np.array(ci_hi)/np.array(nexpected)-1.0
            plt.errorbar(np.ones(nbars),range(nbars),xerr=[xlo,xhi],barsabove=True,fmt=None,ecolor='r')
            
        plt.yticks(range(nbars), terms)
        plt.ylabel('Group')
        plt.xlabel('Representation ratio (95% CI)')
        plt.show()
        
        
        
        
        