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
import matplotlib
import matplotlib.ticker as ticker


def dis2(noi, eps, ls_or_dis, file_name):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org_bc, spc_bc, adv_bc, smt_15_bc, radv_9_bc, org, spc, adv, smt, radv = [], [], [], [], [], [], [], [], [], []

    if ls_or_dis == 'discrepancy':
        for atk in eps:
            org_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='ord') ]['atk_dis'].mean(axis=0))
            spc_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='jcb') ]['atk_dis'].mean(axis=0))
            adv_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='adv') ]['atk_dis'].mean(axis=0))
            radv_9_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='smtadv_9') ]['atk_dis'].mean(axis=0))
            smt_15_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='smtgrad5') ]['atk_dis'].mean(axis=0))
    
    elif ls_or_dis == 'PSNR':        
        for atk in eps:
            org_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='ord') ]['atk_psnr'].mean(axis=0))
            spc_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='jcb') ]['atk_psnr'].mean(axis=0))
            adv_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='adv') ]['atk_psnr'].mean(axis=0))
            radv_9_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='smtadv_9') ]['atk_psnr'].mean(axis=0))
            smt_15_bc.append( df[ (df['atk_eps']==atk) & (df['model_type']=='smtgrad5') ]['atk_psnr'].mean(axis=0))

    fig, ax = plt.subplots(figsize=(5.5,5.3))

    ax.plot(eps, org_bc, color='hotpink',  label='Ord', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot(eps, spc_bc, color='goldenrod',  label='Jcb', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot(eps, radv_9_bc, color='yellowgreen',  label='Adv', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot(eps, adv_bc, color='deepskyblue',  label='Smt-Adv', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot(eps, smt_15_bc, color='orangered',  label='Smt-Grad', alpha=1, linewidth=4.0, markersize=5)#####

    if ls_or_dis == 'discrepancy':
        ax.yaxis.get_offset_text().set_fontsize(18)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_xlabel(r'$\epsilon$', fontsize=18, fontdict=dict(weight='bold'))
    plt.xticks(np.array(eps), fontsize=18)
    plt.yticks(fontsize=20)
    ax.legend(prop={'size': 20})
    ax.set_title(ls_or_dis, fontsize=20)
    img_name = f'plot_bsd68/noise{noi}_sigma0_EPS_{ls_or_dis}.jpg'
    plt.savefig(img_name)

    pdf_name = img_name[:-3] + 'pdf'
    pdf_bytes = img2pdf.convert(img_name)
    file_ = open(pdf_name, "wb")
    file_.write(pdf_bytes)
    file_.close()


def sigma(stds, eps, name, file_name, sig=None):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org_bc, spc_bc, adv_bc, smt_15_bc, radv_9_bc, radv_5_bc, radv_1_bc, smt_1_bc, smt_5_bc, smt_10_bc = [], [], [], [], [], [], [], [], [], []
    org, spc, adv, smt_15, radv_9, radv_5, radv_1, smt_1, smt_5, smt_10 = [], [], [], [], [], [], [], [], [], []
    if name == 'discrepancy':
        org_bc.append( df[ (df['model_type']=='ord') ]['atk_dis'].mean(axis=0) )
        spc_bc.append( df[ (df['model_type']=='jcb') ]['atk_dis'].mean(axis=0) )
        adv_bc.append( df[ (df['model_type']=='adv') ]['atk_dis'].mean(axis=0) )
        radv_1_bc.append( df[ (df['model_type']=='smtadv') ]['atk_dis'].mean(axis=0) )
        smt_1_bc.append( df[ (df['model_type']=='smtgrad') ]['atk_dis'].mean(axis=0) )

        for atk in stds:
            org_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='ord') ]['ub_dis'].values[0] )
            spc_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='jcb') ]['ub_dis'].values[0] )
            adv_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='adv') ]['ub_dis'].values[0] )
            radv_1_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtadv') ]['ub_dis'].values[0] )
            smt_1_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtgrad') ]['ub_dis'].values[0] )
    
    elif name == 'PSNR':  
        org_bc.append( df[ (df['model_type']=='ord') ]['atk_psnr'].mean(axis=0))
        spc_bc.append( df[ (df['model_type']=='jcb') ]['atk_psnr'].mean(axis=0))
        adv_bc.append( df[ (df['model_type']=='adv') ]['atk_psnr'].mean(axis=0))
        radv_1_bc.append( df[ (df['model_type']=='smtadv') ]['atk_psnr'].mean(axis=0))
        smt_1_bc.append( df[ (df['model_type']=='smtgrad') ]['atk_psnr'].mean(axis=0))

        for atk in stds:
            org_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='ord') ]['ub_psnr'].values[0])
            spc_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='jcb') ]['ub_psnr'].values[0])
            adv_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='adv') ]['ub_psnr'].values[0])
            radv_1_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtadv') ]['ub_psnr'].values[0])
            smt_1_bc.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtgrad') ]['ub_psnr'].values[0])


    #error bar
    org_std, spc_std, adv_std, radv_1_std, smt_1_std = [], [], [], [], []
    if name == 'discrepancy':
        org_std.append( df[ (df['model_type']=='ord') ]['atk_dis_std'].mean(axis=0) )
        spc_std.append( df[ (df['model_type']=='jcb') ]['atk_dis_std'].mean(axis=0) )
        adv_std.append( df[ (df['model_type']=='adv') ]['atk_dis_std'].mean(axis=0) )
        radv_1_std.append( df[ (df['model_type']=='smtadv') ]['atk_dis_std'].mean(axis=0) )
        smt_1_std.append( df[ (df['model_type']=='smtgrad') ]['atk_dis_std'].mean(axis=0) )

        for atk in stds:
            org_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='ord') ]['ub_dis_std'].values[0] )
            spc_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='jcb') ]['ub_dis_std'].values[0] )
            adv_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='adv') ]['ub_dis_std'].values[0] )
            radv_1_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtadv') ]['ub_dis_std'].values[0] )
            smt_1_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtgrad') ]['ub_dis_std'].values[0] )
    
    elif name == 'PSNR':  
        org_std.append( df[ (df['model_type']=='ord') ]['atk_psnr_std'].mean(axis=0))
        spc_std.append( df[ (df['model_type']=='jcb') ]['atk_psnr_std'].mean(axis=0))
        adv_std.append( df[ (df['model_type']=='adv') ]['atk_psnr_std'].mean(axis=0))
        radv_1_std.append( df[ (df['model_type']=='smtadv') ]['atk_psnr_std'].mean(axis=0))
        smt_1_std.append( df[ (df['model_type']=='smtgrad') ]['atk_psnr_std'].mean(axis=0))

        for atk in stds:
            org_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='ord') ]['ub_psnr_std'].values[0])
            spc_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='jcb') ]['ub_psnr_std'].values[0])
            adv_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='adv') ]['ub_psnr_std'].values[0])
            radv_1_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtadv') ]['ub_psnr_std'].values[0])
            smt_1_std.append( df[ (df['smooth_std']==atk) & (df['model_type']=='smtgrad') ]['ub_psnr_std'].values[0])


    fig, ax = plt.subplots(figsize=(5.5,5.3)) #5.5,5.3

    ax.plot([0]+stds, org_bc, color='dimgray',  label='Ord', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, spc_bc, color='orange',  label='Jcb', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, adv_bc, color='limegreen',  label='Adv', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, radv_1_bc, color='royalblue',  label='Smt-Adv', alpha=1, linewidth=4.0, markersize=5)#####
    ax.plot([0]+stds, smt_1_bc, color='red',  label='Smt-Grad', alpha=1, linewidth=4.0, markersize=5)#####

    if name == 'discrepancy' or name == 'PSNR':
        ax.errorbar([0]+stds, org_bc, color='dimgray',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(org_std)*0.5, capsize=6)#####
        ax.errorbar([0]+stds, spc_bc, color='orange',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(spc_std)*0.5, capsize=6)#####
        ax.errorbar([0]+stds, adv_bc, color='limegreen',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(adv_std)*0.5, capsize=6)#####
        ax.errorbar([0]+stds, radv_1_bc, color='royalblue',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(radv_1_std)*0.5, capsize=6)#####
        ax.errorbar([0]+stds, smt_1_bc, color='red',  label='_nolegend_', alpha=1, linewidth=2.0, yerr=np.asarray(smt_1_std)*0.5, capsize=6)#####

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
    if 'bsd' in file_name or 'set' in file_name:
        img_name = f'plot_bsd68_dis/noise{eps[0]}_eps{eps[1]}_SIGMA_{name}_NEW.jpg'
    elif 'vk' in file_name:
        img_name = f'plot_vk_dis/noise{eps[0]}_eps{eps[1]}_SIGMA_{name}_NEW.jpg'
    plt.savefig(img_name)

    pdf_name = img_name[:-3] + 'pdf'
    pdf_bytes = img2pdf.convert(img_name)
    file_ = open(pdf_name, "wb")
    file_.write(pdf_bytes)
    file_.close()



if __name__ == '__main__':

    # trend graph for the paper
    sigma([5, 7, 10, 13, 15, 17, 19], (2,10), 'discrepancy', 'bsd68_noi2_eps10')
    sigma([5, 7, 10, 13, 15, 17, 19], (2,10), 'PSNR', 'bsd68_noi2_eps10')

    sigma([5, 7, 10, 13, 15, 17], (10,10), 'discrepancy', 'bsd68_noi10_eps10')
    sigma([5, 7, 10, 13, 15, 17], (10,10), 'PSNR', 'bsd68_noi10_eps10')

    sigma([5, 7, 10, 13, 15, 17], (5,5), 'discrepancy', 'vk_noi5_eps5')
    sigma([5, 7, 10, 13, 15, 17], (5,5), 'PSNR', 'vk_noi5_eps5')

    sigma([5, 7, 10, 13, 15, 17, 19], (5,15), 'discrepancy', 'vk_noi5_eps15')
    sigma([5, 7, 10, 13, 15, 17, 19], (5,15), 'PSNR', 'vk_noi5_eps15')
