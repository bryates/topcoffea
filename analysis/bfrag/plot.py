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
#xsecs = {'ttbar': 830}
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
with uproot.recreate('output.root') as fout:
    for key in output:
        if 'sumw' in key:
            continue #sumw and sumw2 are dicts, not hists
        for s in output[key].axes['dataset']:
            if '_gen' in key:
                continue
            fout[f'histo/{key}_{s}'] = output[key][{'dataset': s, 'systematic': 'nominal'}]


def d0_mass_fit(mass, mean, sigma, nsig, mean_kk, sigma_kk, nkk, mean_pp, sigma_pp, npp, l, nbkg, nbkgg, sigma_bkgg):
    '''
    Define function for fitting D0 mass peak
    Peak Gaussian + exponential bkg + D0 -> KK Gaussian + D0 -> pipi Gaussian
    '''
    return \
    nsig * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma))) + \
    nkk  * np.exp(-1/2 * (np.square(mass - mean_kk) / np.square(sigma_kk))) + \
    npp  * np.exp(-1/2 * (np.square(mass - mean_pp) / np.square(sigma_pp))) + \
    nbkg * np.exp(l * mass)# + \
    #-1 * nbkgg  * np.power((mass - mean) / sigma_bkgg, 2)
    #nbkgg  * np.exp(-1/2 * (np.square(mass - mean) / np.square(sigma_bkgg)))


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
jpsi_mass_bins = np.linspace(2.8, 3.4, 60)
d0mu_mass_bins = np.linspace(1.7, 2.0, 60)
xb_bins = np.linspace(0, 1, 10)
d0mu_xb_bins = np.array([0, .2, .4, .5, .6, .7, .8, .9, 1.])
d0mu_xb_bins = np.linspace(0, 1, 6)

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
    h_samp = [h2[{'dataset': s}] for s in output[f'xb_mass_d0'].axes['dataset'] if 'ttbar' not in s]
    h_name = [s for s in output[f'xb_mass_d0'].axes['dataset'] if 'ttbar' not in s]
    #h2.plot1d(label=f'{meson_tex[meson]}')
    h_samp = h_samp + [h2[{'dataset': 'ttbar'}]]
    h_name = h_name.append('ttbar')
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
    meson = meson.replace('_mu', '')
    meson = meson.replace('mu', '')
    h = output[f'xb_mass_{meson}'][{'dataset': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
    h.plot2d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/xb_mass_{meson_name}.png')
    plt.close()
    h = output[f'xb_mass_{meson}'][{'dataset': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
    xb_bins = h.axes[0].edges
    if 'd0mu' in meson_name:
        xb_bins = d0mu_xb_bins
    print(xb_bins)
    fout = uproot.update('output.root')
    for ibin in range(0,xb_bins.shape[0]-1):
        x = h[{'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum)}]
        for s in output[f'xb_mass_{meson_name}'].axes['dataset']:
            fout[f'histo/xb_mass_{meson_name}_{ibin}_{s}'] = output[f'xb_mass_{meson_name}'][{'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'dataset': s, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
        '''
        if np.sum(x.values()[0]) < 1:
            continue
        '''
        x.plot1d(label=f'{meson_tex[meson]} {ibin}')
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
    h2 = output[f'xb_mass_{meson}'][{'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
    h_name = [s for s in output[f'xb_mass_{meson}'].axes['dataset'] if 'ttbar' not in s]
    h_samp = [(np.sum(h2[{'dataset': s}].values()), i, h2[{'dataset': s}]) for i,s in enumerate(h_name)]
    h_samp.sort(key=lambda x: x[0])
    h_name = [h_name[i] for _,i,_ in h_samp]
    #h2.plot1d(label=f'{meson_tex[meson]}')
    h_samp = [histo for _,_,histo in h_samp]
    h_samp.append(h2[{'dataset': ['ttbar_16', 'ttbar_16APV']}][{'dataset': sum}])
    if 'jpsi' in meson:
        h_samp = [histo[...,::hist.rebin(2)] for histo in h_samp]
    h_name.append('ttbar')
    hep.histplot(h_samp, label=h_name, stack=True, histtype='fill')
    h3 = output[f'xb_mass_{meson}'][{'xb': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal', 'dataset': sum}]
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
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/{meson}_mass_full.png')
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
    h2 = output[f'xb_mass_{meson}'][{'mass': sum, 'meson_id': hist.loc(pdgId), 'systematic': 'nominal'}]
    h_name = [s for s in output[f'xb_mass_{meson}'].axes['dataset'] if 'ttbar' not in s]
    h_samp = [(np.sum(h2[{'dataset': s}].values()), i, h2[{'dataset': s}]) for i,s in enumerate(h_name)]
    h_samp.sort(key=lambda x: x[0])
    h_name = [h_name[i] for _,i,_ in h_samp]
    #h2.plot1d(label=f'{meson_tex[meson]}')
    h_samp = [histo for _,_,histo in h_samp]
    h_samp.append(h2[{'dataset': ['ttbar_16', 'ttbar_16APV']}][{'dataset': sum}])
    h_name.append('ttbar')
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

    h = output[f'vtx_mass_{meson}'][{'dataset': sum, 'systematic': 'nominal'}]
    h.plot2d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/vtx_mass_{meson_name}.png')
    plt.close()

    h = output[f'chi_mass_{meson}'][{'dataset': sum, 'systematic': 'nominal'}]
    h.plot2d()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/chi_mass_{meson_name}.png')
    plt.close()


d0_mean0, d0_sigma0, nd0, kk_mean0, kk_sigma0, nkk, pp_mean0, pp_sigma0, npp, l0, ne0, nbkgg0, bkgg_sigma = 1.87, .01, 0, 1.78, 0.02, 0, 1.9, 0.02, 0, -1, 30, 0, 2*(1.864 - 1.7)
jpsi_mean0, jpsi_sigma0, jpsi_n0, jpsi_alpha0, l0 = 3.097, 0.033, 1, 1.4, -0.5
def plot_and_fit_mass(meson='d0'):
    '''
    Plot and fit the specified meson mass
    '''
    print(f'Fitting {meson}')
    xb_mass = []
    bins = []
    pdgId = mass_id[meson]
    meson_name = meson
    '''
    if 'mu' in meson:
        output[f'xb_mass_{meson}'] = output[f'xb_mass_{meson}'][...,::hist.rebin(2)]
    '''
    meson = meson.replace('_mu', '')
    h = output[f'xb_mass_{meson}'][{'dataset': sum,  'systematic': 'nominal'}]
    fit_func = d0_mass_fit
    mass_bins = d0_mass_bins
    if 'jpsi' in meson:
        mass_bins = jpsi_mass_bins
    elif 'mu' in meson:
        mass_bins = d0mu_mass_bins
    for ibin in range(0,xb_bins.shape[0]-1):
        if ibin < 2 and meson == 'd0': # Little to no signal in first two D0 bins
            xb_mass.append(0)
            bins.append(xb_bins[ibin])
            continue
        x = h[{'xb': slice(hist.loc(xb_bins[ibin]), hist.loc(xb_bins[ibin+1]), sum), 'meson_id': hist.loc(pdgId)}].values()
        ne0 = np.sum(x)#*2
        if ne0 < 21 or ('mu' in meson and ne0<1):
            xb_mass.append(0)
            bins.append(xb_bins[ibin])
            print(f'Warning, not enough events found in {meson} bin {ibin}!')
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
        plt.step(mass_bins, x, label=f'{meson_tex[meson]} {np.round(xb_bins[ibin], 1)} < ' + '$x_{\mathrm{b}}$' + f' < {np.round(xb_bins[ibin+1], 1)}')
        #plt.step(mass_bins[:-1], x, label=f'{meson_tex[meson]} {np.round(xb_bins[ibin], 1)} < ' + '$x_{\mathrm{b}}$' + f' < {np.round(xb_bins[ibin+1], 1)}')
        fit_bounds = ([1.85, 0, 0, 1.77, 0, 0, 1.88, 0, 0, -5, 0, 0, 0.02], [1.88, .02, ne0, 1.79, .05, ne0, 1.91, .02, ne0, 5, ne0, ne0, 10])
        #g = [fit_func(x, *fit_init) for x in mass_bins]
        #plt.plot(mass_bins, g, label=f'Guess {ibin}')
        if 'jpsi' in meson:
            fit_bounds = ([2.8, 0.02, 0, 0, 0, -5, 0], [3.2, 0.05, 2, 5, 10*ne0, 5, ne0])
        try:
            popt, pcov = curve_fit(fit_func, mass_bins, x, p0=fit_init, bounds=fit_bounds)
            #popt, pcov = curve_fit(fit_func, mass_bins[:-1], x, p0=fit_init, bounds=fit_bounds)
        except:
            print(f'Fit {ibin} failed for {meson}!')
            xb_mass.append(0)
            bins.append(xb_bins[ibin])
            continue
        plt.plot(mass_bins, fit_func(mass_bins, *popt), label=f'Fit {ibin}')
        if 'd0' in meson:
            print(f'N D0 {round(popt[2])} +/- {round(np.sqrt(pcov[2][2]))}, N bkg {round(popt[10])} +/- {round(np.sqrt(pcov[10][10]))}')
            xb_mass.append(popt[2])
        elif 'jpsi' in meson:
            print(f'N J/Psi {round(popt[4])} +/- {round(np.sqrt(pcov[4][4]))}, N bkg {round(popt[6])} +/- {round(np.sqrt(pcov[6][6]))}')
            xb_mass.append(popt[4])
        bins.append(xb_bins[ibin])
        chisq = np.sum(np.nan_to_num(np.square(x - fit_func(mass_bins, *popt)) / x, 0, posinf=0, neginf=0))
        #chisq = np.sum(np.nan_to_num(np.square(x - fit_func(mass_bins, *popt)[:-1]) / x, 0, posinf=0, neginf=0))
        print(f'Chi^2 = {chisq} P = {chi2.cdf(chisq, 30)}')
    xb_mass.append(0)
    bins.append(xb_bins[-1])
    plt.legend(ncol=3, bbox_to_anchor=(-0.1, 1.15), loc='upper left', borderaxespad=0.)
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/{meson_name}_mass_fit.png')
    plt.close()
    xb_nom   = output[f'xb_mass_{meson}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId)}].values()
    xb_up    = output[f'xb_mass_{meson}_up'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId)}].values()
    xb_down  = output[f'xb_mass_{meson}_down'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId)}].values()
    plt.step(x=bins, y=xb_mass, label='$x_{\mathrm{b}}$ signal')
    xb_mass_up = xb_mass * (xb_up/np.sum(xb_up)) / (xb_nom/np.sum(xb_nom))
    plt.step(x=bins, y=xb_mass_up, label='$x_{\mathrm{b}}$ signal Up')
    xb_mass_down = xb_mass * (xb_down/np.sum(xb_down)) / (xb_nom/np.sum(xb_nom))
    plt.step(x=bins, y=xb_mass_down, label='$x_{\mathrm{b}}$ signal Down')
    plt.legend()
    hep.cms.label(lumi=lumi_tot)
    plt.savefig(f'{path}/xb_{meson_name}_sig.png')
    plt.close()
    fout = uproot.recreate(f'{meson}_signal.root')
    low  = 2
    high = -1
    bins = bins[low:high]
    nbins = 10
    sig = hist.Hist(hist.axis.Variable(bins + [1.1]),storage=bh.storage.Weight())
    #sig = hist.Hist(hist.axis.Regular(nbins,0,nbins,name='xb'),storage=bh.storage.Weight())
    variance = output[f'xb_mass_{meson}'][{'dataset': sum,  'systematic': 'nominal', 'mass': sum, 'meson_id': hist.loc(pdgId)}].values()
    sig[...] = np.stack([xb_mass[low:high], variance[low:high]], axis=-1)
    print(sig)
    up = hist.Hist(hist.axis.Variable(bins + [1.1]),storage=bh.storage.Weight())
    up[...] = np.stack([xb_mass_up[low:high], variance[low:high]], axis=-1)
    down = hist.Hist(hist.axis.Variable(bins + [1.1]),storage=bh.storage.Weight())
    down[...] = np.stack([xb_mass_down[low:high], variance[low:high]], axis=-1)
    fout['data_obs'] = sig # Asimov hack for now
    fout['xb_sig'] = sig
    fout['xb_sig_up'] = up
    fout['xb_sig_down'] = down
    fout.close()


plot_and_fit_mass('d0')
plot_and_fit_mass('d0mu')
plot_and_fit_mass('jpsi')
#plot_mass('d0')
#plot_mass('jpsi')
#plot_mass('d0mu')
