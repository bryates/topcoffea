import coffea
import coffea.util
import hist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import boost_histogram as bh
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import chi2
import uproot
import mplhep as hep
import json
import argparse

def group(h: hist.Hist, oldname: str, newname: str, grouping: dict[str, list[str]]):
    hnew = hist.Hist(
        hist.axis.StrCategory(grouping, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        storage=h._storage_type,
    )
    for i, indices in enumerate(grouping.values()):
        ind = [c for c in indices if c in h.axes[0]]
        hnew.view(flow=True)[i] = h[{oldname: ind}][{oldname: sum}].view(flow=True)

    return hnew


parser = argparse.ArgumentParser(description='You can select which file to run over')
#parser.add_argument('--xsec', help = 'JSON file of xsecs', required=True)
#parser.add_argument('--lumi', help = 'JSON file of xsecs', default='data/UL/lumi.json')
args = parser.parse_args()

plt.style.use(hep.style.CMS)

output = {}

# Load all pkl files into memory
name = f'histos/coffea_dask.pkl'
fin = coffea.util.load(name)
for key in fin.keys():
    if key in output:
        output[key] += fin[key]
    else:
        output[key] = fin[key]
del fin
print(output['sumw'])
sumw = 0
for dataset in output['sumw']:
    sumw += output['sumw'][dataset]

# Scale processes by cross-section
lumi = 35.9
#xsecs = {'TTToSemiLep': 860}
#j_xsec = open(args.xsec)
#xsecs = json.load(j_xsec)
#j_lumi = open(args.lumi)
#lumi = json.load(j_lumi)
#lumi_tot = np.round(np.sum(list(lumi.values())),1)
lumi_tot = lumi
'''
'''
grouping = {}
g_map = {}
for key in output:
    if 'sumw' in key:
        continue #sumw and sumw2 are dicts, not hists
    for iax,ax in enumerate(list(output[key].axes[0])):
        gkey = ax
        if 'ttW' in ax or 'ttZ' in ax or 'ttz' in ax:
            gkey = 'ttV'
        elif 'tW' in ax:
            gkey = 'tW'
        elif 'SingleT' in ax:
            gkey = 't-ch'
        elif 'DY' in ax:
            gkey = 'DY'
        elif 'W' in ax[0] and 'Jets' in ax:
            gkey = 'WJets'
        elif any([p in ax for p in ['WW', 'WZ', 'ZZ']]):
            gkey = 'Multi boson'
        #else:
        #    gkey = gkey.split('_1')[0]
        g_map[ax] = gkey
        if gkey in grouping:
            grouping[gkey].append(ax)
        else:
            grouping[gkey] = [ax]

for g in grouping:
    grouping[g] = list(set(grouping[g]))

for key in output:
    if 'sumw' in key:
        continue #sumw and sumw2 are dicts, not hists
    for iax,ax in enumerate(list(output[key].axes[0])):
        # Scale all processes by their lumi and the total xsec (testing wth 138 fbinv)
        year = ax.split('_')[1]
        #year_lumi = lumi[year]
        proc = ax#.replace(f'_{year}', '')
        #output[key].view(flow=True)[iax] *= xsecs[proc] / sumw
        #print(proc, g_map[ax], np.sum([output['sumw'][ax] for ax in grouping[g_map[ax]]]))
        #output[key].view(flow=True)[iax] /= np.sum(output['sumw'][ax])
        output[key].view(flow=True)[iax] /= np.sum([output['sumw'][tax] for tax in grouping[g_map[ax]]])
        #output[key].view(flow=True)[iax] /= np.sum([output['sumw'][ax] for ax in grouping[g_map[ax]]])
        #output[key].view(flow=True)[iax] *= xsecs[proc] / np.sum([output['sumw'][ax] for ax in grouping[g_map[ax]]])
        #output[key].view(flow=True)[iax] *= year_lumi * 1000 * xsecs[proc] / np.sum([output['sumw'][ax] for ax in grouping[g_map[ax]]])
        #print(ax, xsecs[proc], [(ax,ax,output['sumw'][ax]) for ax in grouping[g_map[ax]]], np.sum([output['sumw'][ax] for ax in grouping[g_map[ax]]]))
        #output[key].view(flow=True)[iax] *= year_lumi * 1000 * xsecs[ax] / np.sum(output['sumw'][ax])
        gkey = ax
        if 'ttW' in ax or 'ttZ' in ax or 'ttz' in ax:
            gkey = 'ttV'
        elif 'DY' in ax:
            gkey = 'DY'
        elif 'W' in ax[0] and 'Jets' in ax:
            gkey = 'WJets'

for key in output:
    if 'sumw' in key:
        continue #sumw and sumw2 are dicts, not hists
    if output[key].axes['dataset'].value(0) is None:
        continue
    output[key] = group(output[key], 'dataset', 'dataset', grouping)

# Save the total histograms in a pkl file and a ROOT file for testing
coffea.util.save(output, 'coffea.pkl')
'''
with uproot.recreate('output.root') as fout:
    for key in output:
        if 'sumw' in key:
            continue #sumw and sumw2 are dicts, not hists
        for s in output[key].axes['dataset']:
            if '_gen' in key:
                continue
            fout[f'histo/{key}_{s}'] = output[key][{'dataset': s, 'systematic': 'nominal'}]
'''


def d0_mass_fit(mass, mean, sigma, nsig, mean_kk, sigma_kk, nkk, mean_pp, sigma_pp, npp, l, nbkg, nbkgg, sigma_bkgg):
    '''
    Define function for fitting D0 mass peak
    Peak Gaussian + exponential bkg + D0 -> KK Gaussian + D0 -> pipi Gaussian
    '''
    return \
    nsig * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma))) + \
    nkk  * np.exp(-1/2 * (np.square(mass - mean_kk) / np.square(sigma_kk))) + \
    npp  * np.exp(-1/2 * (np.square(mass - mean_pp) / np.square(sigma_pp))) + \
    nbkg * np.exp(l * mass) + \
    -1 * nbkgg  * np.power((mass - mean) / sigma_bkgg, 2)
    #nbkgg  * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma_bkgg)))

def d0mu_mass_fit(mass, mean, sigma, nsig, mean_kk, sigma_kk, nkk, mean_pp, sigma_pp, npp, l, nbkg, nbkgg, sigma_bkgg):
    '''
    Define function for fitting D0 mass peak
    Peak Gaussian + exponential bkg + D0 -> KK Gaussian + D0 -> pipi Gaussian
    '''
    return \
    nsig * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma))) + \
    nkk  * np.exp(-1/2 * (np.square(mass - mean_kk) / np.square(sigma_kk))) + \
    npp  * np.exp(-1/2 * (np.square(mass - mean_pp) / np.square(sigma_pp))) + \
    nbkg * np.exp(l * mass)# + \
    nbkgg  * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma_bkgg)))
    #-1 * nbkgg  * np.power((mass - mean) / sigma_bkgg, 2)


def jpsi_mass_fit(mass, mean, sigma, alpha, n, nsig, l, nbkg):
    '''
    Define function for fitting J/Psi mass peak
    Peak Crystal Ball + exponential bkg
    ''' 
    # Using two Gaussians + expo for now
    return \
    nsig * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma))) + \
    nsig * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma/2))) + \
    nbkg * np.exp(l * mass)
    t = (mass - mean) / sigma
    cb = 0
    if type(mass) == list:
        cb = np.zeros(len(mass))
    cb1 = np.exp(-1/2 * np.square(t))
    a = np.power(n / alpha, n) * np.exp(-1/2 * np.square(alpha))
    #cb2 = np.power(n / alpha, n) * np.exp(-1/2 * np.square(alpha)) / np.power(((n / alpha) - alpha) - t, n)
    b = n / alpha - alpha
    cb2 = a / np.power(b - t, n)
    if type(mass) == list:
        cb[t<-alpha]  = cb2[t<-alpha]
        cb[t>=-alpha] = cb1[t>=-alpha]
    else:
        cb = cb2 if t < -alpha else cb1
         
    '''
    cb = np.exp(-1/2 * np.square(t))
    if t<-alpha:
        cb = np.power(n / alpha, n) * np.exp(-1/2 * np.square(alpha)) / np.power(((n / alpha) - alpha) - t, n)
    '''
    return nsig * cb  + nbkg * np.exp(l * mass)


d0_mass_bins = np.linspace(1.7, 2.0, 60)
jpsi_mass_bins = np.linspace(2.8, 3.4, 30)
d0mu_mass_bins = np.linspace(1.7, 2.0, 30)
xb_bins = np.linspace(0, 1, 10)
d0mu_xb_bins = np.array([0, .2, .4, .5, .6, .7, .8, .9, 1.])
d0mu_xb_bins = np.linspace(0, 1, 6)
d0mu_xb_bins = np.array([0.2, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]) 
d0mu_xb_bins = np.array([0, 0.2, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95]) 

meson_tex = {'d0': '$\mathrm{D^{0}}$', 'd0mu': '$\mathrm{D^{0}}_{\mu}$', 'jpsi': '$\mathrm{J/\psi}$'}
path = '/afs/crc.nd.edu/user/b/byates2/www/BFrag'
path = '/eos/home-b/byates/www/BFrag'

#output['l0pt'].plot1d(label='l0pt')
#plt.legend()
#hep.cms.label(lumi=lumi_tot)
#plt.savefig(f'{path}/l0pt_coffea.png')
#plt.close()
#output['nleps'].plot1d(label='nleps')
#plt.legend()
#hep.cms.label(lumi=lumi_tot)
#plt.savefig(f'{path}/nleps_coffea.png')
#plt.close()
#output['njets'].plot1d(label='njets')
#plt.legend()
#hep.cms.label(lumi=lumi_tot)
#plt.savefig(f'{path}/njets_coffea.png')
#plt.close()
#output['nbjets'].plot1d(label='nbjets')
#plt.legend()
#hep.cms.label(lumi=lumi_tot)
#plt.savefig(f'{path}/nbjets_coffea.png')
#plt.close()
mass_id = {'d0': 421, 'd0mu': 421, 'jpsi': 443}

def plot_mass(meson='d0'):
    '''
    h2 = output[f'xb_mass_d0'][{'xb': sum, 'meson_id': hist.loc(421)}]
    h_samp = [h2[{'dataset': s}] for s in output[f'xb_mass_d0'].axes['dataset'] if 'TTToSemiLep' not in s]
    h_name = [s for s in output[f'xb_mass_d0'].axes['dataset'] if 'TTToSemiLep' not in s]
    #h2.plot1d(label=f'{meson_tex[meson]}')
    h_samp = h_samp + [h2[{'dataset': 'TTToSemiLep'}]]
    h_name = h_name.append('TTToSemiLep')
    print(h_samp, h_name)
    hep.histplot(h_samp, label=h_name, stack=True)
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/d0_mass_full.png')
    plt.close()
    exit()
    Plot and fit the specified meson mass
    '''
    print(f'Plotting {meson}')
    xb_mass = []
    pdgId = mass_id[meson]
    meson_name = meson
    meson = meson.replace('mu', '')
    h = output[f'xb_mass_{meson_name}'][{'dataset': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal', 'jet_flav': sum}]
    h.plot2d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/xb_mass_{meson_name}.png')
    plt.close()
    h = output[f'xb_mass_{meson_name}'][{'dataset': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal', 'jet_flav': sum}]
    xb_bins = h.axes[0].edges
    if 'd0mu' in meson_name:
        xb_bins = d0mu_xb_bins
    print('xb_bins=', xb_bins)
    #fout = uproot.update('output.root')
    for ibin in range(0,xb_bins.shape[0]-1):
        x = h[{'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum)}]
        #for s in output[f'xb_mass_{meson_name}'].axes['dataset']:
            #fout[f'histo/xb_mass_{meson_name}_{ibin}_{s}'] = output[f'xb_mass_{meson_name}'][{'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'dataset': s, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal', 'jet_flav': sum}]
        '''
        if np.sum(x.values()[0]) < 1:
            continue
        '''
        x.plot1d(label=f'{meson_tex[meson_name]} {ibin}')
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/{meson_name}_mass.png')
    plt.close()
    #h1 = output[f'{meson}_mass'][{'meson_id': hist.loc(pdgId)}]
    #h1.plot1d(label=f'{meson_tex[meson]} mass')
    '''
    h2 = output[f'xb_mass_{meson}'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId)}]
    h2.plot1d(label=f'{meson_tex[meson]}')
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/{meson_name}_mass_full.png')
    plt.close()
    '''
    h2 = output[f'xb_mass_{meson_name}'][{'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
    h_name = [s for s in output[f'xb_mass_{meson_name}'].axes['dataset'] if 'TTToSemiLep' not in s and 'TTTo2L2Nu' not in s]
    h_samp = [(np.sum(h2[{'dataset': s}].values()), i, h2[{'dataset': s, 'jet_flav': sum}]) for i,s in enumerate(h_name)]
    h_samp.sort(key=lambda x: x[0])
    h_name = [h_name[i] for _,i,_ in h_samp]
    #h2.plot1d(label=f'{meson_tex[meson]}')
    h_samp = [histo for _,_,histo in h_samp]
    h_samp.append(h2[{'dataset': ['TTToSemiLeptonic_16', 'TTToSemiLeptonic_16APV', 'TTTo2L2Nu_16', 'TTTo2L2Nu_16APV'], 'jet_flav': hist.loc(0)}][{'dataset': sum}])
    #h_samp.append(h2[{'dataset': ['TTToSemiLeptonic_16', 'TTToSemiLeptonic_16APV', 'TTTo2L2Nu_16', 'TTTo2L2Nu_16APV'], 'jet_flav': slice(hist.loc(1), hist.loc(4), sum)}][{'dataset': sum}])
    h_samp.append(h2[{'dataset': ['TTToSemiLeptonic_16', 'TTToSemiLeptonic_16APV', 'TTTo2L2Nu_16', 'TTTo2L2Nu_16APV'], 'jet_flav': hist.loc(4)}][{'dataset': sum}])
    h_samp.append(h2[{'dataset': ['TTTo2L2Nu_16', 'TTTo2L2Nu_16APV'], 'jet_flav': hist.loc(5)}][{'dataset': sum}])
    h_samp.append(h2[{'dataset': ['TTToSemiLeptonic_16', 'TTToSemiLeptonic_16APV'], 'jet_flav': hist.loc(5)}][{'dataset': sum}])
    if 'jpsi' in meson or 'mu' in meson_name:
        h_samp = [histo[...,::hist.rebin(2)] for histo in h_samp]
    h_name.append('$t\overline{t}$ l-jet')
    h_name.append('$t\overline{t}$ c-jet')
    h_name.append('$t\overline{t} \\rightarrow l\overline{l}$')
    h_name.append('$t\overline{t} \\rightarrow l$')
    hep.histplot(h_samp, label=h_name, stack=True, histtype='fill')
    #hep.histplot(h2[{'dataset': sum, 'jet_flav': sum}], stack=True, histtype='fill', sort='yield')
    #hep.histplot(h2[{'dataset': sum, 'jet_flav': sum}], label='Data', histtype='errorbar')
    h3 = output[f'xb_mass_{meson_name}'][{'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal', 'jet_flav': sum, 'dataset': sum}]
    if 'jpsi' in meson or 'mu' in meson_name:
        h3 = h3[...,::hist.rebin(2)]
    hep.histplot(h3, label='Data', histtype='errorbar')
    '''
    h_fsru = output[f'xb_mass_{meson}'][{'xb': sum, 'meson_id': hist.loc(pdgId), 'dataset': sum, 'systematic': 'FSRUp'}]
    h_fsrd = output[f'xb_mass_{meson}'][{'xb': sum, 'meson_id': hist.loc(pdgId), 'dataset': sum, 'systematic': 'FSRDown'}]
    #Fix normalization
    #print(h_fsru)
    #h_fsru *= np.sum(h3.values()) / np.sum(h_fsru)
    #h_fsrd *= np.sum(h3.values()) / np.sum(h_fsrd)
    h_fsru *= np.sum(np.array(list(output['sumw'].values()))) / output['sumwFSRUp']
    #print(h_fsru)
    #hep.histplot(h3, label='Total', stack=True)
    e_fsr = (h_fsru.values() + h_fsrd.values())/2
    err_p = h_fsru.values()[()] # Work around off by one error
    err_m = h_fsrd.values()[()] # Work around off by one error
    bins =  h3.axes[0].edges[:-1]
    print(bins)
    plt.fill_between(bins,err_m,err_p, step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')
    '''
    bins = list(d0_mass_bins)
    error_band_args = { 
        "edges": bins, "facecolor": "none", "linewidth": 0,
        "alpha": .9, "color": "black", "hatch": "///"
    }
    #hep.histplot(h_fsru, label='FSRUp')
    #hep.histplot(h_fsru, label='FSRDown')
    plt.legend(ncol=len(h_name)/3, loc='upper center')
    plt.ylim(0,30000)
    if meson == 'jpsi':
        plt.ylim(0,1100)
    elif meson_name == 'd0mu':
        plt.ylim(0, 1200)
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/{meson_name}_mass_full.png')
    plt.close()

    if meson == 'd0':# and False: #FIXME
        bins = []
        #output['xb_mass_d0_gen'][{'dataset': sum, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'g_id': sum}].plot1d(label='RECO $D^{0}$')
        #for ibin in range(0,xb_bins.shape[0]-1):
        #    unmatch = output['xb_mass_d0_gen'][{'dataset': sum, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'g_id': 0.j}]
        #    for gbin in output['xb_mass_d0_gen'].axes[2].edges:
        #        id_bin = output['xb_mass_d0_gen'].axes[2].value(int(gbin))
        #        if id_bin != 32112 and id_bin != 211321 and id_bin != 0 and id_bin != None:
        #            unmatch += output['xb_mass_d0_gen'][{'dataset': sum, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'g_id': hist.loc(id_bin)}]
        #    h = output['xb_mass_d0_gen'][{'dataset': sum, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'g_id': 211321.j}]
        #    h.plot1d(stack=True, label='$D^{0} \\to \pi K$')
        #    h = output['xb_mass_d0_gen'][{'dataset': sum, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'g_id': 321211.j}]
        #    h.plot1d(stack=True, label='$D^{0} \\to K \pi$')
        #    h = output['xb_mass_d0_gen'][{'dataset': sum, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'g_id': 211211.j}]
        #    h.plot1d(stack=True, label='$D^{0} \\to \pi \pi$')
        #    unmatch.plot1d(stack=True, label='unmatched')
        #fig, ax = plt.subplots(1, 1, figsize=(7,7))
        piK = output['xb_mass_d0_pik'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
        KK = output['xb_mass_d0_kk'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
        pipi = output['xb_mass_d0_pipi'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
        unmatched = output['xb_mass_d0_unmatched'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
        #unmatch = output['xb_mass_d0_gen'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'g_id': 0.j}]
        #output['xb_mass_d0_gen'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'g_id': sum}].plot1d(label='Total $D^{0}$')
        #for gbin in output['xb_mass_d0_gen'].axes[2].edges:
        #    id_bin = output['xb_mass_d0_gen'].axes[2].value(int(gbin))
        #    if id_bin != 32112 and id_bin != 211321 and id_bin != 0 and id_bin != None:
        #        unmatch += output['xb_mass_d0_gen'][{'dataset': sum, 'xb': sum, 'meson_id': hist.loc(pdgId), 'g_id': hist.loc(id_bin)}]
        #unmatch.plot1d(stack=True, label='unmatched')
        hep.histplot([unmatched,pipi,KK,piK], stack=True, label=['Unmatched', '$D^{0} \\to \pi \pi$', '$D^{0} \\to K K$', '$D^{0} \\to \pi K$'], histtype='fill', color=['lightgray', 'red', 'blue', 'green'])
        #hep.histplot([pipi,piK,Kpi], stack=True, label=['$D^{0} \\to \pi \pi$', '$D^{0} \\to \pi K$', '$D^{0} \\to K \pi$'])
        #hep.histplot([unmatch,pipi,piK,Kpi], stack=True, label=['unmatched', '$D^{0} \\to \pi K$', '$D^{0} \\to K \pi$', '$D^{0} \\to \pi \pi$'])
        plt.legend()
        hep.cms.label(lumi=lumi_tot)
        plt.savefig(f'{path}/xb_{meson_name}_gen-match.png')
        plt.close()
    
    
    #output[f'xb_mass_{meson}'][{'mass': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}].plot1d(label='$x_{\mathrm{b}}$')
    #output[f'xb_mass_{meson}'][{'dataset': sum, 'mass': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}].plot1d(label='$x_{\mathrm{b}}$')
    #output[f'xb_mass_{meson}'][{'dataset': sum, f'{meson}_mass': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}].plot1d(label='$x_{\mathrm{b}}$')
    h2 = output[f'xb_mass_{meson_name}'][{'mass': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal', 'jet_flav': sum}]
    h_name = [s for s in output[f'xb_mass_{meson_name}'].axes['dataset'] if 'TTToSemiLep' not in s and 'TTTo2L2Nu' not in s]
    h_samp = [(np.sum(h2[{'dataset': s}].values()), i, h2[{'dataset': s}]) for i,s in enumerate(h_name)]
    h_samp.sort(key=lambda x: x[0])
    h_name = [h_name[i] for _,i,_ in h_samp]
    #h2.plot1d(label=f'{meson_tex[meson]}')
    h_samp = [histo for _,_,histo in h_samp]
    h_samp.append(h2[{'dataset': ['TTToSemiLeptonic_16', 'TTToSemiLeptonic_16APV', 'TTTo2L2Nu_16', 'TTTo2L2Nu_16APV']}][{'dataset': sum}])
    h_name.append('$t\overline{t}$')
    hep.histplot(h_samp, label=h_name, stack=True)
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/xb_{meson_name}_all.png')
    plt.close()
    '''

    output['jet_id'][{'meson_id': hist.loc(pdgId), 'dataset': sum}].plot1d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/jet_id_{meson_name}_all.png')
    plt.close()
    
    h = output[f'ctau'][{'meson_id': hist.loc(mass_id[meson_name])}]
    h.plot1d()
    plt.yscale('log')
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/ctau_{meson_name}.png')
    plt.close()

    if meson != 'd0':
        return
    '''

    '''
    h = output[f'vtx_mass_{meson_name}'][{'dataset': sum, 'systematic': 'nominal'}]
    h.plot2d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/vtx_mass_{meson_name}.png')
    plt.close()

    h = output[f'chi_mass_{meson_name}'][{'dataset': sum, 'systematic': 'nominal'}]
    h.plot2d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/chi_mass_{meson_name}.png')
    plt.close()
    '''


d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, nkk, pp_mean0, pp_sigma0, npp, l0, ne0, nbkgg0, bkgg_sigma = 1.87, .01, 0, 1.78, 0.02, 0, 1.9, 0.02, 0, -1, 60, 0, 2*(1.864 - 1.7)
d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, nkk, pp_mean0, pp_sigma0, npp, l0, ne0, nbkgg0, bkgg_sigma = 1.864, .02, 800, 1.77, 0.02, 0, 1.94, 0.02, 0, -2.7, 180000, 8000, 0.5
jpsi_mean0, jpsi_sigma0, jpsi_n0, jpsi_alpha0, l0 = 3.097, 0.033, 1, 1.4, -0.5
def plot_and_fit_mass(meson='d0'):
    '''
    Plot and fit the specified meson mass
    '''
    print(f'Fitting {meson}')
    xb_data = []
    xb_mass = []
    xb_err = []
    xb_bkg = []
    bins = []
    pdgId = mass_id[meson]
    meson_name = meson
    '''
    if 'mu' in meson:
        output[f'xb_mass_{meson}'] = output[f'xb_mass_{meson}'][...,::hist.rebin(2)]
    '''
    meson = meson.replace('mu', '')
    h = output[f'xb_mass_{meson_name}'][{'systematic': 'nominal'}]#, 'jet_flav': sum}]
    #h = output[f'xb_mass_{meson_name}'][{'dataset': sum,  'systematic': 'nominal', 'jet_flav': sum}]
    fit_func = d0_mass_fit
    mass_bins = d0_mass_bins
    if 'jpsi' in meson:
        mass_bins = jpsi_mass_bins
    elif 'mu' in meson_name:
        mass_bins = d0mu_mass_bins
    xb_bins = np.linspace(0, 1, 10)
    if 'd0mu' in meson_name:
        xb_bins = d0mu_xb_bins
    sig_ds = ['TTToSemiLeptonic_16', 'TTToSemiLeptonic_16APV', 'TTTo2L2Nu_16', 'TTTo2L2Nu_16APV']
    bkg_ds = [s for s in output[f'xb_mass_{meson_name}'].axes['dataset'] if 'TTToSemiLep' not in s and 'TTTo2L2Nu' not in s]
    datasets = {'data': sig_ds + bkg_ds, 'sig': sig_ds, 'bkg': sig_ds + bkg_ds}
    #datasets = {'sig': sig_ds}
    for ibin in range(0,xb_bins.shape[0]-1):
        for ds_name,ds in datasets.items():
            if ds_name == 'sig':
                jets = hist.loc(5) if ds_name == 'sig' else slice(hist.loc(0), hist.loc(5), sum)
            elif ds_name == 'bkg':
                jets = slice(hist.loc(0), hist.loc(5), sum)
            elif ds_name == 'data':
                jets = sum
            if ibin < 2 and meson_name == 'd0': # Little to no signal in first two D0 bins
                if ds_name == 'sig': xb_err.append(0)
                if ds_name == 'sig': xb_mass.append(0)
                elif ds_name == 'data': xb_data.append(0)
                elif ds_name == 'bkg': xb_bkg.append(0)
                if ds_name == 'sig': bins.append(xb_bins[ibin])
                continue
            xb = h[{'dataset': ds, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}][{'dataset': sum}]#[...,::hist.rebin(2)]
            if ds_name == 'bkg':
                xb += h[{'dataset': [s for s in ds if 'TTToSemiLep' not in s and 'TTTo2L2Nu'], 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}][{'dataset': sum}]#[...,::hist.rebin(2)]
            if 'jpsi' in meson_name or 'd0mu' in meson_name:
                xb = h[{'dataset': ds, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}][{'dataset': sum}][...,::hist.rebin(2)]
            #x = h[{'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId)}].values()
            x = xb.values()
            ne0 = np.sum(x)#*2
            if ne0 < 0 or ('mu' in meson and ne0<1):
                if ds_name == 'sig': xb_err.append(0)
                if ds_name == 'sig': xb_mass.append(0)
                elif ds_name == 'data': xb_data.append(0)
                elif ds_name == 'bkg': xb_bkg.append(0)
                if ds_name == 'sig': bins.append(xb_bins[ibin])
                print(f'Warning, not enough events found in {meson_name} bin {ibin}!')
                continue
            nd0 = .001 * ne0
            npp = .1 * nd0*0
            nkk = .1 * nd0*0
            fit_args = [x, d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, nkk, pp_mean0, pp_sigma0, npp, l0, ne0, nbkgg0, bkgg_sigma]
            fit_init = [d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, nkk, pp_mean0, pp_sigma0, npp, l0, ne0, nbkgg0, bkgg_sigma]
            if 'jpsi' in meson:
                fit_func = jpsi_mass_fit
                fit_args = [x, jpsi_mean0, jpsi_sigma0, jpsi_alpha0, jpsi_n0, np.max(x), l0, 0]#.001*ne0]
                fit_init = fit_args[1:]#[jpsi_mean0, jpsi_sigma0, jpsi_n0, jpsi_alpha0, ]
            elif 'mu' in meson_name:
                nd0 = .1 * ne0
                npp = 0
                nkk = 0
                fit_args = [x, d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, nkk, pp_mean0, pp_sigma0, npp, 0, nd0, nbkgg0, bkgg_sigma]
                fit_init = [d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, 0, pp_mean0, pp_sigma0, 0, l0, 0, 0, bkgg_sigma]
                fit_func = d0mu_mass_fit
            plt.errorbar(mass_bins, x, yerr=np.sqrt(x), label=f'{meson_tex[meson_name]} {np.round(xb_bins[ibin], 1)}', fmt='o')
            #plt.errorbar(mass_bins, x, yerr=np.sqrt(x), label=f'{meson_tex[meson]} {np.round(xb_bins[ibin], 1)} < ' + '$x_{\mathrm{b}}$' + f' < {np.round(xb_bins[ibin+1], 1)}', fmt='o')
            #plt.step(mass_bins, x, label=f'{meson_tex[meson]} {np.round(xb_bins[ibin], 1)} < ' + '$x_{\mathrm{b}}$' + f' < {np.round(xb_bins[ibin+1], 1)}')
            #plt.step(mass_bins[:-1], x, label=f'{meson_tex[meson]} {np.round(xb_bins[ibin], 1)} < ' + '$x_{\mathrm{b}}$' + f' < {np.round(xb_bins[ibin+1], 1)}')
            fit_bounds = ([1.85, 0, 0, 1.75, 0.01, 0, 1.9, 0.01, 0, -5, 0, 0, 0.1], [1.87, .02, ne0, 1.85, .05, ne0, 2.0, .05, ne0, 5, ne0, ne0, 5])
            #g = [fit_func(x, *fit_init) for x in mass_bins]
            #plt.plot(mass_bins, g, label=f'Guess {ibin}')
            if 'jpsi' in meson:
                fit_bounds = ([2.8, 0.02, 0, 0, 0, -5, 0], [3.4, 0.05, 2, 5, 10*ne0, 5, ne0])
            elif 'mu' in meson_name:
                fit_bounds = ([1.85, .01, 0, 1.75, 0.01, 0, 1.9, 0.01, 0, -5, 0, 0, 0.1], [1.87, .02, ne0, 1.8, .05, ne0, 2.0, .05, ne0, 5, ne0, ne0, 5])
            try:
                popt, pcov = curve_fit(fit_func, mass_bins, x, p0=fit_init, bounds=fit_bounds)
                #popt, pcov = curve_fit(fit_func, mass_bins[:-1], x, p0=fit_init, bounds=fit_bounds)
            except:
                print(f'Fit {ibin} {ds_name} failed for {meson_name}!')
                if ds_name == 'sig': xb_err.append(0)
                if ds_name == 'sig': xb_mass.append(0)
                elif ds_name == 'data': xb_data.append(0)
                elif ds_name == 'bkg': xb_bkg.append(0)
                if ds_name == 'sig': bins.append(xb_bins[ibin])
                continue
            #plt.plot(xb.axes[0].edges[:-1], fit_func(mass_bins, *popt), label=f'Fit {ibin}')
            b = np.linspace(mass_bins[0], mass_bins[-1], 600)
            plt.plot(b, fit_func(b, *popt),  label=f'Fit {ibin}')
            if 'd0' in meson:
                print(f'N D0 {round(popt[2])} +/- {round(np.sqrt(pcov[2][2]))}, N bkg {round(popt[5] + popt[8] + popt[10] + popt[11])} +/- {round(np.sqrt(pcov[10][10]))}')
                if ds_name == 'sig': xb_mass.append(popt[2])
                elif ds_name == 'data': xb_data.append(popt[2])
                #xb_mass.append(popt[2] / (xb_bins[ibin+1] - xb_bins[ibin]))
                elif ds_name == 'bkg': xb_bkg.append(popt[2])
                #elif ds_name == 'bkg': xb_bkg.append(popt[5] + popt[8] + popt[10] + popt[11])
            elif 'jpsi' in meson:
                print(f'N J/Psi {round(popt[4])} +/- {round(np.sqrt(pcov[4][4]))}, N bkg {round(popt[6])} +/- {round(np.sqrt(pcov[6][6]))}')
                if ds_name == 'sig': xb_mass.append(popt[4])
                elif ds_name == 'data': xb_data.append(popt[4])
                elif ds_name == 'bkg': xb_bkg.append(popt[4])
                #elif ds_name == 'bkg': xb_bkg.append(popt[6])
            if ds_name == 'sig': xb_err.append(np.trace(pcov))
            if ds_name == 'sig': bins.append(xb_bins[ibin])
            chisq = np.sum(np.nan_to_num(np.square(x - fit_func(mass_bins, *popt)) / x, 0, posinf=0, neginf=0))
            #chisq = np.sum(np.nan_to_num(np.square(x - fit_func(mass_bins, *popt)[:-1]) / x, 0, posinf=0, neginf=0))
            print(f'Chi^2 = {chisq} P = {chi2.cdf(chisq, 60)}')
    xb_err.append(0)
    xb_mass.append(0)
    xb_data.append(0)
    xb_bkg.append(0)
    bins.append(xb_bins[-1])
    if meson_name == 'd0':
        plt.ylim(0, 10000)
    elif meson_name == 'jpsi':
        plt.ylim(0, 300)
    elif meson_name == 'd0mu':
        plt.ylim(0, 300)
    plt.legend(ncol=4, loc='upper left', borderaxespad=0.)
    #plt.legend(ncol=4, bbox_to_anchor=(-0.1, 1.15), loc='upper left', borderaxespad=0.)
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/{meson_name}_mass_fit.png')
    plt.close()
    for ibin in range(0,xb_bins.shape[0]-1):
        for ds_name,ds in datasets.items():
            if ds_name == 'sig':
                jets = hist.loc(5) if ds_name == 'sig' else slice(hist.loc(0), hist.loc(5), sum)
            elif ds_name == 'bkg':
                jets = slice(hist.loc(0), hist.loc(5), sum)
            elif ds_name == 'data':
                jets = sum
            xb = h[{'dataset': ds, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}][{'dataset': sum}]#[...,::hist.rebin(2)]
            if ds_name == 'bkg':
                xb += h[{'dataset': [s for s in ds if 'TTToSemiLep' not in s and 'TTTo2L2Nu'], 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}][{'dataset': sum}]#[...,::hist.rebin(2)]
                print([{'dataset': [s for s in ds if 'TTToSemiLep' not in s and 'TTTo2L2Nu' not in s], 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}])
            if 'jpsi' in meson_name or 'd0mu' in meson_name:
                xb = h[{'dataset': ds, 'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId), 'jet_flav': jets}][{'dataset': sum}][...,::hist.rebin(2)]
            x = xb.values()
            plt.errorbar(mass_bins, x, yerr=np.sqrt(x), label=f'{meson_tex[meson_name]} {np.round(xb_bins[ibin], 1)}', fmt='o')
            try:
                popt, pcov = curve_fit(fit_func, mass_bins, x, p0=fit_init, bounds=fit_bounds)
                b = np.linspace(mass_bins[0], mass_bins[-1], 600)
            except:
                plt.close()
                print(f'Fit {ibin} {ds_name} failed for {meson_name}!')
                continue
            plt.plot(b, fit_func(b, *popt),  label=f'Fit {ibin}')
            if ds_name == 'sig': plt.savefig(f'{path}/{meson_name}_mass_fit_{ibin}.png')
            elif ds_name == 'bkg': plt.savefig(f'{path}/{meson_name}_mass_bkg_fit_{ibin}.png')
            elif ds_name == 'bkg': print(f'{path}/{meson_name}_mass_bkg_fit_{ibin}.png')
            plt.close()
    xb_bkg_c = output[f'xb_mass_{meson_name}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(4)}].values()
    xb_bkg_l = output[f'xb_mass_{meson_name}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(0)}].values()
    xb   = output[f'xb_mass_{meson_name}_nom'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(5)}].values()
    xb_nom   = output[f'xb_mass_{meson_name}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(5)}].values()
    #xb_nom   = output[f'xb_mass_{meson_name}_nom'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(5)}].values()
    xb_up    = output[f'xb_mass_{meson_name}_up'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(5)}].values()
    xb_down  = output[f'xb_mass_{meson_name}_down'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': hist.loc(5)}].values()
    #xb_nom -= xb_bkg_c
    #xb_nom -= xb_bkg_l
    #xb_up -= xb_bkg_c
    #xb_up -= xb_bkg_l
    #xb_down -= xb_bkg_c
    #xb_down -= xb_bkg_l
    #plt.step(x=bins, y=xb_bkg_l, label='$x_{\mathrm{b}}$ l-jet')
    #plt.step(x=bins, y=xb_bkg_c, label='$x_{\mathrm{b}}$ c-jet')
    #xb_up *= np.sum(xb_nom) / np.sum(xb_up)
    #xb_down *= np.sum(xb_nom) / np.sum(xb_down)
    print(len(xb_mass), xb_mass)
    print(len(xb_up), xb_up)
    print(len(xb_down), xb_down)
    xb_up = np.nan_to_num(xb_mass * (xb_up/np.sum(xb_up)) / (xb_nom/np.sum(xb_nom)), nan=0)
    xb_down = np.nan_to_num(xb_mass * (xb_down/np.sum(xb_down)) / (xb_nom/np.sum(xb_nom)), nan=0)
    xb_mass = np.nan_to_num(xb_mass * (xb/np.sum(xb)) / (xb_nom/np.sum(xb_nom)), nan=0)
    bkg = xb_bkg_l + xb_bkg_c
    bkg = xb_nom - xb_mass
    #xb_up -= bkg
    #xb_down -= bkg
    low  = 2
    high = -1
    if 'd0mu' in meson_name:
        low  = 0
        high = -1
        xb_err[0] = 0
        xb_err[1] = 0
    nbins = 10
    xb_tot = h[{'dataset': sum, 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': sum}].values()
    data_obs = output[f'xb_mass_{meson_name}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': sum}][low:nbins+high] # Asimov hack for now
    edg = data_obs.axes[0].edges
    if meson_name == 'd0mu':
        edg = d0mu_xb_bins
    #tmp = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    width = np.array([x - y for x,y in zip(edg[1:], edg[:-1])])
    #width = tmp.axes[0].widths
    #plt.step(x=bins, y=xb_nom, label='$x_{\mathrm{b}}$ nom')
    #plt.step(x=bins, y=bkg, label='$x_{\mathrm{b}}$ bkg')
    #plt.step(x=bins, y=xb_mass+bkg, label='$x_{\mathrm{b}}$ tot')
    plt.step(x=bins, y=xb_mass, label='$x_{\mathrm{b}}$ signal')
    xb_mass_up = xb_up
    xb_mass_down = xb_down
    #xb_mass_up = np.nan_to_num(xb_mass * (xb_up/np.sum(xb_up)) / (xb_nom/np.sum(xb_nom)), nan=0)
    #xb_mass_up = xb_mass_up / width
    plt.step(x=bins, y=xb_mass_up, label='$x_{\mathrm{b}}$ signal Up')
    #xb_mass_down = np.nan_to_num(xb_mass * (xb_down/np.sum(xb_down)) / (xb_nom/np.sum(xb_nom)), nan=0)
    #xb_mass_down = xb_mass_down / width
    plt.step(x=bins, y=xb_mass_down, label='$x_{\mathrm{b}}$ signal Down')
    plt.step(x=bins, y=xb_bkg, label='$x_{\mathrm{b}}$ bkg')
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/xb_{meson_name}_sig.png')
    plt.close()
    fout = uproot.recreate(f'{meson_name}_signal.root')
    #if 'jpsi' in meson: low = 3
    bins = bins[low:high]
    xb_axis  = hist.axis.Regular(name="xb",   label="$x_{\mathrm{b}}$", bins=10, start=0, stop=1)
    tot = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    bkg = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    bkg_c = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    bkg_l = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    data = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    sig = hist.Hist(hist.axis.Variable(edg ),storage=bh.storage.Weight())
    tot = hist.Hist(hist.axis.Variable(edg ))
    #sig = hist.Hist(hist.axis.Regular(nbins,0,nbins,name='xb'),storage=bh.storage.Weight())
    variance = output[f'xb_mass_{meson_name}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId), 'jet_flav': sum}].values()
    #bkg[...] = np.stack([xb_bkg[low:high], variance[low:high]], axis=-1)
    #sig[...] = np.stack([xb_mass[low:high], variance[low:high]], axis=-1)
    #bkg[...] = np.stack([xb_bkg[low:high], xb_bkg[low:high]], axis=-1)
    #xb_bkg = xb_tot - xb_mass
    tot[...] = xb_tot[low:high]
    bkg[...] = np.stack([xb_bkg[low:high], xb_bkg[low:high]], axis=-1)
    bkg_c[...] = np.stack([xb_bkg[low:high], xb_bkg_c[low:high]], axis=-1)
    bkg_l[...] = np.stack([xb_bkg[low:high], xb_bkg_l[low:high]], axis=-1)
    #sig[...] = np.stack([xb_nom[low:high], xb_err[low:high]], axis=-1)
    data[...] = np.stack([xb_data[low:high], xb_data[low:high]], axis=-1)
    sig[...] = np.stack([xb_mass[low:high], xb_err[low:high]], axis=-1)
    xb_mass_up *= np.sum(xb_mass) / np.sum(xb_mass_up)
    xb_mass_down *= np.sum(xb_mass) / np.sum(xb_mass_down)
    up = hist.Hist(hist.axis.Variable(edg),storage=bh.storage.Weight())
    print(xb_mass, xb_err)
    #up[...] = np.stack([xb_mass_up[low:high], variance[low:high]], axis=-1)
    up[...] = np.stack([xb_mass_up[low:high], xb_err[low:high]], axis=-1)
    print('xb_mass', xb_mass)
    print('xb_mass_up', xb_mass_up)
    print('xb_nom', xb_nom)
    print('xb_up', xb_up)
    down = hist.Hist(hist.axis.Variable(edg),storage=bh.storage.Weight())
    #down[...] = np.stack([xb_mass_down[low:high], variance[low:high]], axis=-1)
    down[...] = np.stack([xb_mass_down[low:high], xb_err[low:high]], axis=-1)
    #print(data_obs - sig)
    print('tot')
    print(tot)
    print('tot - sig')
    print(tot - sig)
    print('bkg')
    print(bkg)
    print('data')
    print(data)
    print('sig')
    print(sig)
    print('up')
    print(up)
    print('down')
    print(down)
    fout['data_obs'] = data
    fout['xb_bkg'] = bkg
    fout['d_bkg'] = tot - sig
    #fout['xb_bkg_c'] = bkg_c#data_obs - sig
    #fout['xb_bkg_l'] = bkg_l#data_obs - sig
    #fout['data_obs'] = sig
    #fout['xb_bkg'] = data_obs - sig
    fout['xb_sig'] = sig
    fout['xb_sig_up'] = up
    fout['xb_sig_down'] = down
    fout.close()


plot_and_fit_mass('d0')
plot_and_fit_mass('d0mu')
plot_and_fit_mass('jpsi')
plot_mass('d0')
plot_mass('jpsi')
plot_mass('d0mu')
