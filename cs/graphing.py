#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os 
import pandas as pd
import math
import img2pdf
import matplotlib.ticker as ticker



def sigma(stds, eps, cs, b, name, file_name='test_dec18', sup=None):
    
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org, spc, adv1, rad1, smt1 = [], [], [], [], []
    org_std, spc_std, adv1_std, rad1_std, smt1_std = [], [], [], [], []

    if name == 'discrepancy':
        org.append( df[ (df['model_type']=='ISTA') ]['atk_dis'].mean(axis=0) )
        spc.append( df[ (df['model_type']=='JCBgma20') ]['atk_dis'].mean(axis=0) )
        adv1.append( df[ (df['model_type']=='ADV') ]['atk_dis'].mean(axis=0) )
        rad1.append( df[ (df['model_type']=='smtADV') ]['atk_dis'].mean(axis=0) )
        smt1.append( df[ (df['model_type']=='STH') ]['atk_dis'].mean(axis=0) )
        for std in stds:
            org.append( df[ (df['smooth_std']==std) & (df['model_type']=='ISTA') ]['ub_dis'].values[0] )
            spc.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['ub_dis'].values[0] )
            adv1.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['ub_dis'].values[0] )
            rad1.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['ub_dis'].values[0] )
            smt1.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['ub_dis'].values[0] )

    elif name == 'PSNR':
        org.append( df[ (df['model_type']=='ISTA') ]['atk_psnr'].mean(axis=0) )
        spc.append( df[ (df['model_type']=='JCBgma20') ]['atk_psnr'].mean(axis=0) )
        adv1.append( df[ (df['model_type']=='ADV') ]['atk_psnr'].mean(axis=0) )
        rad1.append( df[ (df['model_type']=='smtADV') ]['atk_psnr'].mean(axis=0) )
        smt1.append( df[ (df['model_type']=='STH') ]['atk_psnr'].mean(axis=0) )
        for std in stds:
            org.append( df[ (df['smooth_std']==std) & (df['model_type']=='ISTA') ]['ub_psnr'].values[0] )
            spc.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['ub_psnr'].values[0] )
            adv1.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['ub_psnr'].values[0] )
            rad1.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['ub_psnr'].values[0] )
            smt1.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['ub_psnr'].values[0] )


    #error bar
    if name == 'discrepancy':
        org_std.append( df[ (df['model_type']=='ISTA') ]['atk_dis_std'].mean(axis=0) )
        spc_std.append( df[ (df['model_type']=='JCBgma20') ]['atk_dis_std'].mean(axis=0) )
        adv1_std.append( df[ (df['model_type']=='ADV') ]['atk_dis_std'].mean(axis=0) )
        rad1_std.append( df[ (df['model_type']=='smtADV') ]['atk_dis_std'].mean(axis=0) )
        smt1_std.append( df[ (df['model_type']=='STH') ]['atk_dis_std'].mean(axis=0) )
        for std in stds:
            org_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='ISTA') ]['ub_dis_std'].values[0] )
            spc_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['ub_dis_std'].values[0] )
            adv1_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['ub_dis_std'].values[0] )
            rad1_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['ub_dis_std'].values[0] )
            smt1_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['ub_dis_std'].values[0] )

    elif name == 'PSNR':
        org_std.append( df[ (df['model_type']=='ISTA') ]['atk_psnr_std'].mean(axis=0) )
        spc_std.append( df[ (df['model_type']=='JCBgma20') ]['atk_psnr_std'].mean(axis=0) )
        adv1_std.append( df[ (df['model_type']=='ADV') ]['atk_psnr_std'].mean(axis=0) )
        rad1_std.append( df[ (df['model_type']=='smtADV') ]['atk_psnr_std'].mean(axis=0) )
        smt1_std.append( df[ (df['model_type']=='STH') ]['atk_psnr_std'].mean(axis=0) )
        for std in stds:
            org_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='ISTA') ]['ub_psnr_std'].values[0] )
            spc_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['ub_psnr_std'].values[0] )
            adv1_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['ub_psnr_std'].values[0] )
            rad1_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['ub_psnr_std'].values[0] )
            smt1_std.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['ub_psnr_std'].values[0] )

    fig, ax = plt.subplots(figsize=(5.5,5.3))

    ax.plot([0]+stds, org, color='dimgray',  label='Ord', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, spc, color='orange',  label='Jcb', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, adv1, color='limegreen',  label='Adv', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, rad1, color='royalblue',  label='Smt-Adv', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, smt1, color='red',  label='Smt-Grad', alpha=1, linewidth=4.0, markersize=5)#####

    if name == 'discrepancy' or name == 'PSNR':
        scale = 0.5
        ax.errorbar([0]+stds, org, color='dimgray',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(org_std)*scale, capsize=6)#####
        ax.errorbar([0]+stds, spc, color='orange',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(spc_std)*scale, capsize=6)#####
        ax.errorbar([0]+stds, adv1, color='limegreen',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(adv1_std)*scale, capsize=6)#####
        ax.errorbar([0]+stds, rad1, color='royalblue',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(rad1_std)*scale, capsize=6)#####
        ax.errorbar([0]+stds, smt1, color='red',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(smt1_std)*scale, capsize=6)#####

    if name == 'discrepancy':
        ax.yaxis.get_offset_text().set_fontsize(18)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_xlabel(r'$\sigma$', fontsize=18, fontdict=dict(weight='bold'))
    plt.xticks(np.array([0]+stds), fontsize=18)
    if name == 'PSNR':
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.yticks(fontsize=20)
    if name == 'discrepancy':
        ax.legend(prop={'size': 20})#, loc='lower left')
    ax.set_title(name, fontsize=20)
    if 'set' in file_name:
        img_name = f'plot_set/eps{eps}_cs{cs}_b{b}_SIGMA_{name}.jpg'
    else:
        img_name = f'plot_ig/eps{eps}_cs{cs}_b{b}_SIGMA_{name}.jpg'
    plt.savefig(img_name)

    pdf_name = img_name[:-3] + 'pdf'
    pdf_bytes = img2pdf.convert(img_name)
    file_ = open(pdf_name, "wb")
    file_.write(pdf_bytes)
    file_.close()



if __name__ == '__main__':

    #trends images for paper
    sigma(stds=[5, 9, 13, 17, 21, 25, 29, 33], eps=5, cs=10, b=0, name='PSNR', file_name='setmay_e5_c10_n0', sup=True)
    sigma(stds=[5, 9, 13, 17, 21, 25, 29, 33], eps=5, cs=10, b=0, name='discrepancy', file_name='setmay_e5_c10_n0', sup=True)

    sigma(stds=[3,5,7,9,11,13,20,25], eps=5, cs=30, b=0, name='PSNR', file_name='setmay_e5_c30_n0', sup=True)
    sigma(stds=[3,5,7,9,11,13,20,25], eps=5, cs=30, b=0, name='discrepancy', file_name='setmay_e5_c30_n0', sup=True)

    sigma(stds=[3, 5, 7, 9, 13, 20, 25, 29, 33], eps=3, cs=10, b=0, name='PSNR', file_name='igmay_e3_c10_n0', sup=True)
    sigma(stds=[3, 5, 7, 9, 13, 20, 25, 29, 33], eps=3, cs=10, b=0, name='discrepancy', file_name='igmay_e3_c10_n0', sup=True)

    sigma(stds=[5, 10, 15, 25, 30, 35, 40, 45], eps=7, cs=10, b=0, name='PSNR', file_name='igmay_e7_c10_n0', sup=True)
    sigma(stds=[5, 10, 15, 25, 30, 35, 40, 45], eps=7, cs=10, b=0, name='discrepancy', file_name='igmay_e7_c10_n0', sup=True)
