#!/usr/bin/env python
import copy
import coffea
import uproot
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import hist
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.GetValuesFromJsons import get_param, get_lumi
from topcoffea.modules.objects import *
from topcoffea.modules.corrections import GetBTagSF, ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, GetPUSF, ApplyRochesterCorrections, ApplyJetSystematics, AttachPSWeights, AttachScaleWeights, GetTriggerSF
from topcoffea.modules.selection import *
from topcoffea.modules.paths import topcoffea_path


# Takes strings as inputs, constructs a string for the full channel name
# Try to construct a channel name like this: [n leptons]_[lepton flavors]_[p or m charge]_[on or off Z]_[n b jets]_[n jets]
    # chan_str should look something like "3l_p_offZ_1b", NOTE: This function assumes nlep comes first
    # njet_str should look something like "atleast_5j",   NOTE: This function assumes njets comes last
    # flav_str should look something like "emm"
def construct_cat_name(chan_str,njet_str=None,flav_str=None):

    # Get the component strings
    nlep_str = chan_str.split("_")[0] # Assumes n leps comes first in the str
    chan_str = "_".join(chan_str.split("_")[1:]) # The rest of the channel name is everything that comes after nlep
    if chan_str == "": chan_str = None # So that we properly skip this in the for loop below
    if flav_str is not None:
        flav_str = flav_str
    if njet_str is not None:
        njet_str = njet_str[-2:] # Assumes number of n jets comes at the end of the string
        if "j" not in njet_str:
            # The njet string should really have a "j" in it
            raise Exception(f"Something when wrong while trying to consturct channel name, is \"{njet_str}\" an njet string?")

    # Put the component strings into the channel name
    ret_str = nlep_str
    for component in [flav_str,chan_str,njet_str]:
        if component is None: continue
        ret_str = "_".join([ret_str,component])
    return ret_str


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._dtype = dtype

        # Create the histograms
        self.jpsi_mass_bins = np.linspace(2.8, 3.4, 60)
        self.d0_mass_bins = np.linspace(1.7, 2.0, 60)
        self.xb_bins = np.linspace(0, 1, 10)
        self.systematics = ['nominal', 'FSRup', 'FSRdown', 'ISRup', 'ISRdown']

        dataset_axis = hist.axis.StrCategory(name="dataset", label="", categories=[], growth=True)
        ht_axis = hist.axis.Regular(name="ht", label="$\it{H}_{\mathrm{T}}$ [GeV]", bins=100, start=0, stop=1000)
        met_axis = hist.axis.Regular(name="met", label="$\it{p}_{\mathrm{T}}^{\mathrm{miss}}$ [GeV]", bins=30, start=0, stop=300)
        j_pt_ch_axis = hist.axis.Regular(name="j_pt_ch", label="Leading jet $\Sigma p^{\mathrm{ch}_{\mathrm{T}}$ [GeV]", bins=30, start=0, stop=300)
        jpt_axis = hist.axis.Regular(name="j0pt", label="Leading jet $\it{p}_{\mathrm{T}}$ [GeV]", bins=30, start=0, stop=300)
        bpt_axis = hist.axis.Regular(name="b0pt", label="Leading b jet $\it{p}_{\mathrm{T}}$ [GeV]", bins=30, start=0, stop=300)
        lpt_axis = hist.axis.Regular(name="l0pt", label="Leading lepton $\it{p}_{\mathrm{T}}$ [GeV]", bins=30, start=0, stop=300)
        D0pt_axis= hist.axis.Regular(name="D0pt", label="Leading D0 $\it{p}_{\mathrm{T}}$ [GeV]", bins=10, start=0, stop=100)
        D0pipt_axis= hist.axis.Regular(name="D0pipt", label="Leading D0 pi $\it{p}_{\mathrm{T}}$ [GeV]", bins=10, start=0, stop=100)
        D0kpt_axis= hist.axis.Regular(name="D0kpt", label="Leading D0 k $\it{p}_{\mathrm{T}}$ [GeV]", bins=10, start=0, stop=100)
        xb_jpsi_axis  = hist.axis.Regular(name="xb_jpsi",   label="$x_{\mathrm{b}}$", bins=10, start=0, stop=1)
        xb_axis  = hist.axis.Regular(name="xb",   label="$x_{\mathrm{b}}$", bins=10, start=0, stop=1)
        d0mu_xb_axis  = hist.axis.Variable(np.array([0, 0.2, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]), name="xb",   label="$x_{\mathrm{b}}$")
        xb_ch_axis  = hist.axis.Regular(name="xb_ch",   label="$x_{\mathrm{b}} \Sigma it{p}_{\mathrm{T}}^{\mathrm{charged}}$", bins=10, start=0, stop=1)
        pdgid_axis= hist.axis.Regular(name="pdgid",   label="D0 id's", bins=10, start=0, stop=250)
        d0_axis  = hist.axis.Regular(name='d0',   label="$d_0$", bins=10, start=0, stop=100)
        njets_axis = hist.axis.Regular(name='njets', label='$N_{\mathrm{jets}}$', bins=10, start=0, stop=10)
        nbjets_axis = hist.axis.Regular(name='nbjets', label='$N_{\mathrm{b-jets}}$', bins=10, start=0, stop=10)
        nleps_axis = hist.axis.Regular(name='nleps', label='$N_{\mathrm{leps}}$', bins=10, start=0, stop=10)
        jpsi_mass_axis = hist.axis.Regular(name='mass', label='J/Psi mass [GeV]', bins=len(self.jpsi_mass_bins), start=self.jpsi_mass_bins[0], stop=self.jpsi_mass_bins[-1])
        d0_mass_axis = hist.axis.Regular(name='mass', label='D0 mass [GeV]', bins=len(self.d0_mass_bins), start=self.d0_mass_bins[0], stop=self.d0_mass_bins[-1])
        mass_axes = [hist.axis.Regular(name=f'd0_{int(xb_bin*10)}', label='D0 mass [GeV] (' + str(round(self.xb_bins[ibin], 2)) + ' < $x_{\mathrm{b}}$ < ' + str(round(self.xb_bins[ibin+1], 2)) + ')', bins=len(self.d0_mass_bins), start=self.d0_mass_bins[0], stop=self.d0_mass_bins[-1]) for ibin,xb_bin in enumerate(self.xb_bins[:-1])]
        meson_axis = hist.axis.IntCategory(name="meson_id", label="Meson pdgId (421 D0, 443 J/Psi, 443211 J/Psi+K)", categories=[421, 443, 42113], growth=True)
        #meson_axis = hist.axis.IntCategory(name="meson_id", label="Meson pdgId (421 D0, 42113 D0mu, 443 J/Psi, 443211 J/Psi+K)", categories=[421, 42113, 443])
        # light = 0
        # c = 4
        # b = 5
        flav_axis = hist.axis.IntCategory(name="jet_flav", label="Jet hadron flavor", categories=[0, 4, 5], growth=True)
        jet_axis = hist.axis.IntCategory(name="jet_id", label="Jet flavor", categories=list(range(1,7)))
        pi_gen_axis = hist.axis.IntCategory(name="pi_gid", label="Gen-matched flavor", categories=[], growth=True)
        k_gen_axis = hist.axis.IntCategory(name="k_gid", label="Gen-matched flavor", categories=[], growth=True)
        pi_mother_gen_axis = hist.axis.IntCategory(name="pi_mother", label="Gen-matched flavor", categories=[], growth=True)
        k_mother_gen_axis = hist.axis.IntCategory(name="k_mother", label="Gen-matched flavor", categories=[], growth=True)
        cos2D_axis = hist.axis.Regular(name='cos2D', label='Decay opening angle', bins=100, start=-1, stop=1)
        nmeson_axis = hist.axis.Regular(name='nmeson', label='Decay opening angle', bins=10, start=0, stop=10)
        ctau_axis = hist.axis.Regular(name='ctau', label='Meson time of flight significance', bins=110, start=-10, stop=100)
        cut_axis = hist.axis.StrCategory(name='cut', categories=[],growth=True)
        systematic_axis = hist.axis.StrCategory(name='systematic', categories=['nominal'],growth=True)
        self.output = processor.dict_accumulator({
            'ht': hist.Hist(dataset_axis, meson_axis, ht_axis, cut_axis, systematic_axis),
            'met': hist.Hist(dataset_axis, meson_axis, met_axis, cut_axis, systematic_axis),
            'j_pt_ch': hist.Hist(dataset_axis, meson_axis, j_pt_ch_axis, cut_axis, systematic_axis),
            'j0pt': hist.Hist(dataset_axis, meson_axis, jpt_axis, cut_axis, systematic_axis),
            'b0pt': hist.Hist(dataset_axis, meson_axis, bpt_axis, cut_axis, systematic_axis),
            'l0pt': hist.Hist(dataset_axis, meson_axis, lpt_axis, cut_axis, systematic_axis),
            'D0pt': hist.Hist(dataset_axis, meson_axis, D0pt_axis, cut_axis, systematic_axis),
            'D0pipt': hist.Hist(dataset_axis, meson_axis, D0pipt_axis, cut_axis, systematic_axis),
            'D0kpt': hist.Hist(dataset_axis, meson_axis, D0kpt_axis, cut_axis, systematic_axis),
            'jet_id'  : hist.Hist(dataset_axis, meson_axis, jet_axis, cut_axis, systematic_axis),
            'xb_mass_jpsi'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0mu'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_jpsi_nom'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0mu_nom'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0_nom'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_jpsi_up'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0mu_up'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0_up'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_jpsi_down'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0mu_down'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0_down'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, flav_axis, cut_axis, systematic_axis),
            'xb_mass_d0_pik'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, cut_axis, systematic_axis),
            'xb_mass_d0_kk'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, cut_axis, systematic_axis),
            'xb_mass_d0_pipi'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, cut_axis, systematic_axis),
            'xb_mass_d0_unmatched'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, cut_axis, systematic_axis),
            'xb_mass_d0_gen'  : hist.Hist(dataset_axis, meson_axis, pi_gen_axis, k_gen_axis, pi_mother_gen_axis, k_mother_gen_axis, xb_axis, d0_mass_axis, cut_axis, systematic_axis),
            'xb_jpsi'  : hist.Hist(dataset_axis, xb_jpsi_axis, cut_axis, systematic_axis),
            'xb_d0'  : hist.Hist(dataset_axis, xb_axis, cut_axis, systematic_axis),
            'xb_d0mu'  : hist.Hist(dataset_axis, xb_axis, cut_axis, systematic_axis),
            'xb_ch'  : hist.Hist(dataset_axis, xb_ch_axis, cut_axis, systematic_axis),
            'pdgid'  : hist.Hist(dataset_axis, pdgid_axis, cut_axis, systematic_axis),
            'd0'  : hist.Hist(dataset_axis, d0_axis, cut_axis, systematic_axis),
            'njets' : hist.Hist(dataset_axis, meson_axis, njets_axis, cut_axis, systematic_axis),
            'nbjets' : hist.Hist(dataset_axis, meson_axis, nbjets_axis, cut_axis, systematic_axis),
            'nleps' : hist.Hist(dataset_axis, meson_axis, nleps_axis, cut_axis, systematic_axis),
            'jpsi_mass': hist.Hist(dataset_axis, meson_axis, jpsi_mass_axis, cut_axis, systematic_axis),
            'd0_mass': hist.Hist(dataset_axis, meson_axis, d0_mass_axis, cut_axis, systematic_axis),
            'cos2D': hist.Hist(dataset_axis, cos2D_axis, meson_axis, cut_axis, systematic_axis),
            'nmeson': hist.Hist(dataset_axis, nmeson_axis, meson_axis, cut_axis, systematic_axis),
            'ctau': hist.Hist(dataset_axis, ctau_axis, meson_axis, flav_axis, cut_axis, systematic_axis),
            'nvtx' : hist.Hist(dataset_axis, meson_axis, hist.axis.Regular(name='nvtx', label='N vtx', bins=60, start=0, stop=60), cut_axis, systematic_axis),
            'vtx_mass_d0' : hist.Hist(dataset_axis, hist.axis.Regular(name='vtx', label='Vertex prob.', bins=100, start=0, stop=.1), d0_mass_axis, cut_axis, systematic_axis),
            'chi_mass_d0' : hist.Hist(dataset_axis, hist.axis.Regular(name='chi', label='$\chi^2$ vtx', bins=100, start=0, stop=5), d0_mass_axis, cut_axis, systematic_axis),
            'vtx_mass_jpsi' : hist.Hist(dataset_axis, hist.axis.Regular(name='vtx', label='Vertex prob.', bins=100, start=0, stop=.1), jpsi_mass_axis, cut_axis, systematic_axis),
            'chi_mass_jpsi' : hist.Hist(dataset_axis, hist.axis.Regular(name='chi', label='$\chi^2$ vtx', bins=100, start=0, stop=5), jpsi_mass_axis, cut_axis, systematic_axis),
            'sumw'  : processor.defaultdict_accumulator(float),
            'sumw2' : processor.defaultdict_accumulator(float),
            #'sumw_syst' : hist.Hist(hist.axis.Regular(name='weight', label='weight', bins=2, start=0, stop=2), systematic_axis),
        })

        # Set the list of hists to fill
        if hist_lst is None:
            # If the hist list is none, assume we want to fill all hists
            self._hist_lst = list(self.output.keys())
        else:
            # Otherwise, just fill the specified subset of hists
            for hist_to_include in hist_lst:
                if hist_to_include not in self.output.keys():
                    raise Exception(f"Error: Cannot specify hist \"{hist_to_include}\", it is not defined in the processor.")
            self._hist_lst = hist_lst # Which hists to fill

        # Set the energy threshold to cut on
        self._ecut_threshold = ecut_threshold

        # Set the booleans
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories


    @property
    def accumulator(self):
        return self.output

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata["dataset"]

        isData             = self._samples[dataset]["isData"]
        year               = self._samples[dataset]['year']#'20' + dataset.split('_')[-1]
        xsec               = self._samples[dataset]["xsec"]
        sow                = np.ones_like(events["event"])#self._samples[dataset]["nSumOfWeights"]
        frag = uproot.open(topcoffea_path('../analysis/bfrag/bfragweights.root'))

        # Get up down weights from input dict
        '''
        if (self._do_systematics and not isData):
            if histAxisName in get_param("lo_xsec_samples"):
                # We have a LO xsec for these samples, so for these systs we will have e.g. xsec_LO*(N_pass_up/N_gen_nom)
                # Thus these systs will cover the cross section uncty and the acceptance and effeciency and shape
                # So no NLO rate uncty for xsec should be applied in the text data card
                sow_ISRUp          = self._samples[dataset]["nSumOfWeights"]
                sow_ISRDown        = self._samples[dataset]["nSumOfWeights"]
                sow_FSRUp          = self._samples[dataset]["nSumOfWeights"]
                sow_FSRDown        = self._samples[dataset]["nSumOfWeights"]
                sow_renormUp       = self._samples[dataset]["nSumOfWeights"]
                sow_renormDown     = self._samples[dataset]["nSumOfWeights"]
                sow_factUp         = self._samples[dataset]["nSumOfWeights"]
                sow_factDown       = self._samples[dataset]["nSumOfWeights"]
                sow_renormfactUp   = self._samples[dataset]["nSumOfWeights"]
                sow_renormfactDown = self._samples[dataset]["nSumOfWeights"]
            else:
                # Otherwise we have an NLO xsec, so for these systs we will have e.g. xsec_NLO*(N_pass_up/N_gen_up)
                # Thus these systs should only affect acceptance and effeciency and shape
                # The uncty on xsec comes from NLO and is applied as a rate uncty in the text datacard
                sow_ISRUp          = self._samples[dataset]["nSumOfWeights_ISRUp"          ]
                sow_ISRDown        = self._samples[dataset]["nSumOfWeights_ISRDown"        ]
                sow_FSRUp          = self._samples[dataset]["nSumOfWeights_FSRUp"          ]
                sow_FSRDown        = self._samples[dataset]["nSumOfWeights_FSRDown"        ]
                sow_renormUp       = self._samples[dataset]["nSumOfWeights_renormUp"       ]
                sow_renormDown     = self._samples[dataset]["nSumOfWeights_renormDown"     ]
                sow_factUp         = self._samples[dataset]["nSumOfWeights_factUp"         ]
                sow_factDown       = self._samples[dataset]["nSumOfWeights_factDown"       ]
                sow_renormfactUp   = self._samples[dataset]["nSumOfWeights_renormfactUp"   ]
                sow_renormfactDown = self._samples[dataset]["nSumOfWeights_renormfactDown" ]
        '''
        if (self._do_systematics and not isData):
            sow_ISRUp          = 1
            sow_ISRDown        = 1
            sow_FSRUp          = 1
            sow_FSRDown        = 1
            sow_renormUp       = 1
            sow_renormDown     = 1
            sow_factUp         = 1
            sow_factDown       = 1
            sow_renormfactUp   = 1
            sow_renormfactDown = 1
        else:
            sow_ISRUp          = -1
            sow_ISRDown        = -1
            sow_FSRUp          = -1
            sow_FSRDown        = -1
            sow_renormUp       = -1
            sow_renormDown     = -1
            sow_factUp         = -1
            sow_factDown       = -1
            sow_renormfactUp   = -1
            sow_renormfactDown = -1

        datasets = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        #for d in datasets:
        #    if d in dataset: dataset = dataset.split('_')[0]
        if any(d in dataset for d in datasets): dataset = dataset.split('_')[1]

        # Set the sampleType (used for MC matching requirement)
        sampleType = "prompt"
        if isData:
            sampleType = "data"

        # Initialize objects
        met  = events.MET
        ele  = events.Electron
        mu   = events.Muon
        jets = events.Jet
        # Charm meson candidates from b-jets
        charm_cand = events.BToCharm
        #ptsort = ak.argsort(charm_cand.fit_pt, ascending=False)
        ptsort = ak.local_index(charm_cand.fit_pt)
        charm_cand = events.BToCharm[ptsort]
        #jets = jets[charm_cand.jetIdx]

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(events.MET.pt)

        #ele = ele[isPresElec(ele.pt, ele.eta, ele.dxy, ele.dz, ele.miniPFRelIso_all, ele.sip3d, getattr(ele,"mvaFall17V2noIso_WPL"))]
        #mu = mu[isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)]
        ele["isTightLep"] = isTightElec(ele)
        mu["isTightLep"]  = isTightMuon(mu)
        ele["isVetoLep"]  = isVetoElec(ele)
        mu ["isVetoLep"]  = isVetoMuon(mu)
        ele = ele[isTightElec(ele)]
        mu  = mu[isTightMuon(mu)]
        #ele = ele[isTightElec(ele) & ~isVetoElec(ele)]
        #mu  = mu[isTightMuon(mu) & ~isVetoMuon(mu)]
        AttachElectronSF(ele,year=year)
        AttachMuonSF(mu,year=year)
        leptons = ak.with_name(ak.concatenate([ele, mu], axis=1), 'PtEtaPhiMCandidate')
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
        leptons = mu
        events["leptons"] = leptons
        llpairs = ak.combinations(leptons, 2, fields=["l0","l1"])
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)
        #events["minMllSFOS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)
        add1lMaskAndSFs(events, year, isData, 'prompt')

        # Initialize the out object
        hout = self.output

        # Get the lumi mask for data
        if year == "2016" or year == "2016APV":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        elif year == "2017":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        elif year == "2018":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

        ######### Systematics ###########

        # Define the lists of systematics we include
        obj_correction_syst_lst = [
            f'JER_{year}Up',f'JER_{year}Down', # Systs that affect the kinematics of objects
            'JES_FlavorQCDUp', 'JES_AbsoluteUp', 'JES_RelativeBalUp', 'JES_BBEC1Up', 'JES_RelativeSampleUp', 'JES_FlavorQCDDown', 'JES_AbsoluteDown', 'JES_RelativeBalDown', 'JES_BBEC1Down', 'JES_RelativeSampleDown'
        ]
        wgt_correction_syst_lst = [
            #"PUUp","PUDown", # Exp systs
            #f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown", # Exp systs
            "lepSF_muonUp","lepSF_muonDown",f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown","PreFiringUp","PreFiringDown",f"triggerSF_{year}Up",f"triggerSF_{year}Down", # Exp systs
            #"lepSF_muonUp","lepSF_muonDown","lepSF_elecUp","lepSF_elecDown",f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown","PreFiringUp","PreFiringDown",f"triggerSF_{year}Up",f"triggerSF_{year}Down", # Exp systs
            "FSRUp","FSRDown","ISRUp","ISRDown" # Theory systs
        ]
        data_syst_lst = [
        ]

        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        lumi = 1
        if not isData:

            # Get the genWeight
            genw = np.nan_to_num(events.genWeight, nan=0, posinf=0, neginf=0)

            # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
            # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
            lumi = 1000.0*get_lumi(year)
            weights_obj_base.add("norm",(xsec/sow)*genw*lumi)

            # Attach PS weights (ISR/FSR) and scale weights (renormalization/factorization) and PDF weights
            AttachPSWeights(events)
            #AttachScaleWeights(events)
            #AttachPdfWeights(events) # TODO
            # FSR/ISR weights
            weights_obj_base.add('ISR', events.nom, events.ISRUp*(sow/sow_ISRUp), events.ISRDown*(sow/sow_ISRDown))
            weights_obj_base.add('FSR', events.nom, events.FSRUp*(sow/sow_FSRUp), events.FSRDown*(sow/sow_FSRDown))
            # renorm/fact scale
            #weights_obj_base.add('renorm', events.nom, events.renormUp*(sow/sow_renormUp), events.renormDown*(sow/sow_renormDown))
            #weights_obj_base.add('fact', events.nom, events.factUp*(sow/sow_factUp), events.factDown*(sow/sow_factDown))
            # Prefiring and PU (note prefire weights only available in nanoAODv9)
            weights_obj_base.add('PreFiring', events.L1PreFiringWeight.Nom,  events.L1PreFiringWeight.Up,  events.L1PreFiringWeight.Dn)
            weights_obj_base.add('PU', GetPUSF((events.Pileup.nTrueInt), year), GetPUSF(events.Pileup.nTrueInt, year, 'up'), GetPUSF(events.Pileup.nTrueInt, year, 'down'))


        ######### The rest of the processor is inside this loop over systs that affect object kinematics  ###########

        # If we're doing systematics and this isn't data, we will loop over the obj_correction_syst_lst list
        if self._do_systematics and not isData: syst_var_list = ["nominal"] + obj_correction_syst_lst
        # Otherwise loop juse once, for nominal
        else: syst_var_list = ['nominal']

        # Loop over the list of systematic variations we've constructed
        met_raw=met
        for syst_var in syst_var_list:
            # Make a copy of the base weights object, so that each time through the loop we do not double count systs
            # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics
            weights_obj_base_for_kinematic_syst = copy.deepcopy(weights_obj_base)

            #################### Jets ####################

            # Jet cleaning, before any jet selection
            #vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
            lep = ak.with_name(ak.concatenate([ele, mu], axis=1), 'PtEtaPhiMCandidate')
            vetos_tocleanjets = ak.with_name( lep, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            jets['isClean'] = ~ak.any(tmp.slot0 == tmp.slot1, axis=-1) # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index
            cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

            # Selecting jets and cleaning them
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            # Jet energy corrections
            if not isData:
                cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
                cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
                cleanedJets["pt_gen"] =ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
                cleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, cleanedJets.pt)[0]
                events_cache = events.caches[0]
                cleanedJets = ApplyJetCorrections(year, corr_type='jets').build(cleanedJets, lazy_cache=events_cache)
                # SYSTEMATICS
                cleanedJets=ApplyJetSystematics(year,cleanedJets,syst_var)
                met=ApplyJetCorrections(year, corr_type='met').build(met_raw, cleanedJets, lazy_cache=events_cache)
            jets["isGood"] = isTightJet(jets.pt, jets.eta, jets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
            cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
            goodJets = cleanedJets[cleanedJets.isGood]

            # Count jets
            njets = ak.num(goodJets)
            ht = ak.sum(goodJets.pt,axis=-1)
            #ht = ak.sum(jets[jets.isGood & jets.isClean].pt,axis=-1)
            j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]
            j0pt = ak.fill_none(ak.pad_none(ak.sort(goodJets.pt, axis=1, ascending=False), 1), 0)
            #j0pt = ak.fill_none(ak.pad_none(ak.sort(jets.pt, axis=1, ascending=False), 1), 0)
            nj   = ak.num(ak.fill_none(ak.pad_none(ak.sort(goodJets.pt, axis=1, ascending=False), 1), 0))
            l0pt = ak.fill_none(ak.pad_none(ak.sort(leptons.pt, axis=1, ascending=False), 1), 0)
            nl   = ak.num(ak.fill_none(ak.pad_none(ak.sort(leptons.pt, axis=1, ascending=False), 1), 0))

            # Loose DeepJet WP
            if year == "2017":
                btagwpl = get_param("btag_wp_loose_UL17")
            elif year == "2018":
                btagwpl = get_param("btag_wp_loose_UL18")
            elif year=="2016":
                btagwpl = get_param("btag_wp_loose_UL16")
            elif year=="2016APV":
                btagwpl = get_param("btag_wp_loose_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])
            addTriggerCat(events)
            # Move to selection.py
            def is_ttbar(jets, bjets, mu, ele, leptons, met):
                jets_pt = ak.fill_none(ak.pad_none(jets.pt, 1), 0)
                leps_pt  = ak.fill_none(ak.pad_none(leptons.pt, 1), 0)
                
                jets_eta = ak.fill_none(ak.pad_none(jets.eta, 1), 0)
                leps_eta  = ak.fill_none(ak.pad_none(leptons.eta, 1), 0)

                #llpairs = ak.fill_none(ak.pad_none(ak.combinations(leptons, 2, fields=["l0","l1"]), 1), False)
                #llpairs = llpairs[~ak.is_none(llpairs)]
                
                #goodJet  = ak.fill_none(ak.pad_none(jets.isGood & jets.isClean, 1), False)
                #goodBJet = ak.fill_none(ak.pad_none(bjets.isGood & bjets.isClean, 1), False)
                #nJets = ak.num(jets[goodJet])
                #nBJets = ak.num(bjets[goodBJet])
                nJets = ak.num(jets)
                nBJets = ak.num(bjets)
                nLep = ak.num(leptons)
                jpt30 = jets_pt[:,0] > 30
                jpt30 = ak.max(jets.pt, -1) > 30
                bpt30 = ak.max(bjets.pt, -1) > 30
                #nmu = ak.num(mu, axis=-1)
                #nele = ak.num(ele, axis=-1)
                #DY = leptons.pt > 0 # Make a mask of all True
                #if ak.any(llpairs):
                #    DY[(nmu + nele)>1][(abs(llpairs.l0.pdgId)==abs(llpairs.l1.pdgId))] = met.pt > 40 # Suppress DY in same-flavor dilepton
                #QCD = leptons.pt > 0 # Make a mask of all True
                #if ak.any(llpairs):
                #    QCD[(nmu + nele)>1] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1) > 20 # Suppress QDD
                #lpt = leptons.pt < 0 # Make a mask of all False
                #if ak.any((nmu + nele) > 1):
                #    lpt[((nmu + nele)>1) & (abs(leptons.pdgId)==13)] = mu.pt > 20 # Muons in dilepton events
                #    lpt[((nmu + nele)>1) & (abs(leptons.pdgId)==11)] = ele.pt > 30 # Electrons in dilepton events
                #lpt[(nmu == 1) & (nele == 0)] = mu.pt > 24 # Muons in single lepton events
                #lpt[(nmu == 0) & (nele == 1)] = ele.pt > 35 # Electrons in single lepton events
                jeta24 = np.abs(jets_eta) < 2.4
                leta24 = np.abs(leps_eta) < 2.4
                jeta24 = np.abs(jets.eta[ak.argmax(jets.pt, axis=-1, keepdims=True)]) < 2.4
                leta24 = np.abs(leptons.eta[ak.argmax(leptons.pt, axis=-1, keepdims=True)]) < 2.4
                lpt25  = ak.fill_none(ak.pad_none(leptons.pt > 25, 1), False)
                lpt20  = ak.fill_none(ak.pad_none(leptons.pt > 20, 1), False)
                lpt25 = ak.num(lpt25) == 1
                lpt20 = ak.num(lpt20) > 1
                lpt = lpt25 ^ lpt20
                #lpt = ak.fill_none(ak.firsts(leptons.pt) < 0, False)
                #lpt = ak.where([nLep[lpt25] == 1], lpt25, lpt)
                #print(lpt, '\n\n\n\n')
                #print((lpt & (nLep[lpt20] > 1)) | (lpt & (nLep[lpt25] == 1)), '\n\n\n\n')
                #is_ttbar = (ak.num(jets)>4)&(ak.num(bjets_tight)>1)&((ak.num(ele)>1)|(ak.num(mu)>1))# & (pad_jets[:,0].pt>30)# & ((pad_ele[:,0].pt>25) | (pad_mu[:,0].pt>25))#&(ht>180)
                #return QCD & DY & (nJets >= 1) & (nBJets >=1) & (nLep >= 1) & bpt30 & jpt30 & lpt# & jeta24 & leta24
                #FIXME
                return ((nJets >= 3) & (nBJets >= 1) & (nLep >= 1), # & ak.any(goodJet, axis=-1) & ak.any(goodBJet, axis=-1)
                       (nJets >= 3) & (nBJets == 0) & (nLep >= 1))# & ak.any(goodJet, axis=-1) & ak.any(goodBJet, axis=-1)
                #return (nJets >= 3) & (nBJets >=1) & (nLep >= 1)# & ak.any(goodJet, axis=-1) & ak.any(goodBJet, axis=-1)
                #return (nJets >= 1) & (nBJets >=1) & (nLep >= 1) & lpt & ak.any(goodJet, axis=-1) & ak.any(goodBJet, axis=-1)
                #return (nJets >= 1) & (nBJets >=1) & (nLep >= 1) * ak.any(goodJet, axis=-1) & ak.any(goodBJet, axis=-1)

            # Medium DeepJet WP
            if year == "2017":
                btagwpm = get_param("btag_wp_medium_UL17")
            elif year == "2018":
                btagwpm = get_param("btag_wp_medium_UL18")
            elif year=="2016":
                btagwpm = get_param("btag_wp_medium_UL16")
            elif year=="2016APV":
                btagwpm = get_param("btag_wp_medium_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])

            # Tight DeepJet WP
            if year == "2017":
                btagwpt = get_param("btag_wp_tight_UL17")
            elif year == "2018":
                btagwpt = get_param("btag_wp_tight_UL18")
            elif year=="2016":
                btagwpt = get_param("btag_wp_tight_UL16")
            elif year=="2016APV":
                btagwpt = get_param("btag_wp_tight_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            #isBtagJetsTight = (goodJets.btagDeepFlavB > btagwpt)
            goodJet = jets[charm_cand.jetIdx].isGood & jets[charm_cand.jetIdx].isClean
            #isBtagJetsTight = (jets[charm_cand.jetIdx].btagDeepFlavB > btagwpt)
            isBtagJetsTight = (goodJets.btagDeepFlavB > btagwpt)
            isNotBtagJetsTight = np.invert(isBtagJetsTight)

            #bjets_tight = jets[isBtagJetsTight & jets.isGood & jets.isClean]
            bjets_tight = goodJets[isBtagJetsTight]
            #bjets_tight = jets[charm_cand.jetIdx][isBtagJetsTight]
            #bjets_tight = bjets_tight[bjets_tight.isGood & bjets_tight.isClean]
            b0pt = ak.sort(bjets_tight.pt, axis=1, ascending=False)
            nbj  = ak.num(ak.fill_none(ak.pad_none(ak.sort(bjets_tight.pt, axis=1, ascending=False), 1), 0))
            #goodJet = jets[charm_cand.jetIdx].isGood
            events['is_ttbar'] = is_ttbar(goodJets, bjets_tight, mu, ele, mu, met)[0]
            events['is_qcd'] = is_ttbar(goodJets, bjets_tight, mu, ele, mu, met)[1]
            #events['is_ttbar'] = is_ttbar(goodJets, bjets_tight, mu, ele, leptons, met)


            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets

            ######### Event weights that do not depend on the lep cat ##########

            if not isData:

                # Btag SF following 1a) in https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
                bJetSF   = GetBTagSF(goodJets, year, 'TIGHT')
                bJetEff  = GetBtagEff(goodJets, year, 'tight')
                bJetEff_data   = bJetEff*bJetSF
                pMC     = ak.prod(bJetEff       [isBtagJetsTight], axis=-1) * ak.prod((1-bJetEff       [~isBtagJetsTight]), axis=-1)
                pMC     = ak.where(pMC==0,1,pMC) # removeing zeroes from denominator...
                pData   = ak.prod(bJetEff_data  [isBtagJetsTight], axis=-1) * ak.prod((1-bJetEff_data  [~isBtagJetsTight]), axis=-1)
                weights_obj_base_for_kinematic_syst.add("btagSF", pData/pMC)

                if self._do_systematics and syst_var=='nominal':
                    for b_syst in ["bc_corr","light_corr",f"bc_{year}",f"light_{year}"]:
                        bJetSFUp = GetBTagSF(goodJets, year, 'TIGHT', syst=b_syst)[0]
                        bJetSFDo = GetBTagSF(goodJets, year, 'TIGHT', syst=b_syst)[1]
                        bJetEff_dataUp = bJetEff*bJetSFUp
                        bJetEff_dataDo = bJetEff*bJetSFDo
                        pDataUp = ak.prod(bJetEff_dataUp[isBtagJetsTight], axis=-1) * ak.prod((1-bJetEff_dataUp[~isBtagJetsTight]), axis=-1)
                        pDataDo = ak.prod(bJetEff_dataDo[isBtagJetsTight], axis=-1) * ak.prod((1-bJetEff_dataDo[~isBtagJetsTight]), axis=-1)
                        weights_obj_base_for_kinematic_syst.add(f"btagSF{b_syst}", events.nom, (pDataUp/pMC)/(pData/pMC),(pDataDo/pMC)/(pData/pMC))

                # Trigger SFs
                GetTriggerSF(year,events,leptons)
                weights_obj_base_for_kinematic_syst.add(f"triggerSF_{year}", events.trigger_sf, copy.deepcopy(events.trigger_sfUp), copy.deepcopy(events.trigger_sfDown))            # In principle does not have to be in the lep cat loop


            ######### Event weights that do depend on the lep cat ###########

            # Loop over categories and fill the dict
            weights_dict = {}
            for ch_name in ["ttbar", "d0", "d0mu", "jpsi", "qcd"]:

                # For both data and MC
                weights_dict[ch_name] = copy.deepcopy(weights_obj_base_for_kinematic_syst)

                # For MC only
                if not isData:
                    weights_dict[ch_name].add("lepSF_muon", events.sf_1l_muon, copy.deepcopy(events.sf_1l_hi_muon), copy.deepcopy(events.sf_1l_lo_muon))
                    #weights_dict[ch_name].add("lepSF_elec", events.sf_1l_elec, copy.deepcopy(events.sf_1l_hi_elec), copy.deepcopy(events.sf_1l_lo_elec))
                '''
                    if ch_name.startswith("2l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_2l_muon, copy.deepcopy(events.sf_2l_hi_muon), copy.deepcopy(events.sf_2l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_2l_elec, copy.deepcopy(events.sf_2l_hi_elec), copy.deepcopy(events.sf_2l_lo_elec))
                    elif ch_name.startswith("3l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_3l_muon, copy.deepcopy(events.sf_3l_hi_muon), copy.deepcopy(events.sf_3l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_3l_elec, copy.deepcopy(events.sf_3l_hi_elec), copy.deepcopy(events.sf_3l_lo_elec))
                    elif ch_name.startswith("4l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_4l_muon, copy.deepcopy(events.sf_4l_hi_muon), copy.deepcopy(events.sf_4l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_4l_elec, copy.deepcopy(events.sf_4l_hi_elec), copy.deepcopy(events.sf_4l_lo_elec))
                    else:
                        raise Exception(f"Unknown channel name: {ch_name}")
                '''


            ######### Masks we need for the selection ##########

            # Pass trigger mask
            pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

            # Charm meson candidates from b-jets
            ctau_mask = ak.fill_none((charm_cand.vtx_l3d / charm_cand.vtx_el3d) > 10, False)
            ctau_mask = ak.fill_none(ak.pad_none((charm_cand.vtx_l3d / charm_cand.vtx_el3d) > 10, 1), False)
            ctau = charm_cand.vtx_l3d / charm_cand.vtx_el3d
            chi2_mask = ak.fill_none((charm_cand.svprob>0.02) & (charm_cand.chi2/charm_cand.ndof<5), False)
            chi2_mask = charm_cand.chi2/charm_cand.ndof<5
            chi2_mask = ak.fill_none(ak.pad_none(charm_cand.chi2/charm_cand.ndof<5, 1), False)
            chi2sort = ak.argsort(charm_cand.chi2/charm_cand.ndof, ascending=True)
            #ptsort = ak.argmax(charm_cand.fit_pt, axis=-1)
            ht_mask = ht > 180
            b_mask = goodJet & (jets[charm_cand.jetIdx].btagDeepFlavB > btagwpt)
            b_mask = ak.fill_none(ak.pad_none(goodJet & (jets[charm_cand.jetIdx].btagDeepFlavB > btagwpt), 1), False)
            #b_mask = goodJets[charm_cand.jetIdx].btagDeepFlavB > btagwpt
            ptzmask = get_Z_peak_mask(leptons,15,flavor="os")
            cand_mask = ak.fill_none(ak.pad_none(charm_cand.jetIdx>-1, 1), False)
            meson_id = ak.fill_none(ak.pad_none(copy.deepcopy(np.abs(charm_cand.meson_id)), 1), 0)
            d0_mask = ak.fill_none(ak.pad_none(meson_id==421, 1), False)
            #d0_mask = ak.fill_none(ak.pad_none(np.abs(charm_cand.meson_id) == 421, 1), False)
            d0_mask_411 = ak.fill_none(ak.pad_none(meson_id==411, 1), False)
            #d0_mask_411 = ak.fill_none(ak.pad_none(np.abs(charm_cand.meson_id) == 411, 1), False)
            d0_mask = events.is_ttbar & b_mask & ht_mask & ctau_mask & chi2_mask & cand_mask & d0_mask # FIXME typo data files, used 411 instead of 421
            d0_mask_411 = events.is_ttbar & b_mask & ht_mask & ctau_mask & chi2_mask & cand_mask & d0_mask_411 # FIXME typo data files, used 411 instead of 421
            if isData:
                d0_mask = d0_mask_411
            mcount = ak.num(meson_id)
            meson_id = ak.flatten(meson_id)
            meson_id = ak.where(meson_id==411, 421, meson_id)
            #meson_id = ak.where(ak.flatten(d0_mask), 421, meson_id)
            meson_id = ak.unflatten(meson_id, mcount)
            d0_mask = ak.fill_none(ak.pad_none(meson_id==421, 1), False)
            d0_mask = events.is_ttbar & b_mask & ht_mask & ctau_mask & chi2_mask & cand_mask & d0_mask # FIXME typo data files, used 411 instead of 421
            ctau_low_mask = ak.fill_none(ak.pad_none((charm_cand.vtx_l3d / charm_cand.vtx_el3d) > 2, 1), False)
            jpsi_mask = ak.fill_none(ak.pad_none(np.abs(charm_cand.meson_id) == 443, 1), False)
            jpsi_mask = events.is_ttbar & chi2_mask & ctau_low_mask & cand_mask & jpsi_mask
            #d0_mask = events.is_ttbar & ptzmask & ht_mask & ctau_mask & chi2_mask & (charm_cand.jetIdx>-1) & (np.abs(charm_cand.meson_id) == 421)
            #jpsi_mask = events.is_ttbar & ptzmask & ctau_mask & chi2_mask & (charm_cand.jetIdx>-1) & (np.abs(charm_cand.meson_id) == 443)
            mass = charm_cand.fit_mass

            # D0 mu tagged
            pi_gidx = ak.firsts(ak.fill_none(charm_cand.pigIdx, 0))
            k_gidx = ak.firsts(ak.fill_none(charm_cand.kgIdx, 0))
            
            pi_gid = ak.firsts(ak.fill_none(charm_cand.pigId, 0))
            k_gid = ak.firsts(ak.fill_none(charm_cand.kgId, 0))
            pi_mother = ak.firsts(ak.fill_none(charm_cand.pi_mother, 0))
            k_mother = ak.firsts(ak.fill_none(charm_cand.k_mother, 0))
            #d0_gid = np.abs(pi_gid)*1e6 + np.abs(k_gid)*1e3 + np.abs)
            maskpik = ((abs(pi_gid)==211) & (abs(k_gid)==321))
            maskkk = ((abs(pi_gid)==321) & (abs(k_gid)==321))
            maskpipi = ((abs(pi_gid)==211) & (abs(k_gid)==211))

            # D0 mu tagged
            d0_mu_mask = ak.fill_none(ak.pad_none(np.abs(charm_cand.meson_id) == 421, 1), False)
            d0_mu_mask_411 = ak.fill_none(ak.pad_none(np.abs(charm_cand.meson_id) == 411, 1), False)
            mu_mask = ak.fill_none(ak.pad_none(np.abs(charm_cand.x_id)==13, 1), False)
            d0_mu_mask = events.is_ttbar & b_mask & ctau_mask & chi2_mask & cand_mask & d0_mu_mask & mu_mask
            d0_mu_mask_411 = events.is_ttbar & b_mask & ctau_mask & chi2_mask & cand_mask & d0_mu_mask_411 & mu_mask
            if isData:
                d0_mu_mask = d0_mu_mask_411
            #d0_mu_mask = d0_mu_mask | d0_mu_mask_411
            d0_mu_mask = ptzmask & (np.abs(charm_cand.x_id)==13)
            d0_mu_mask = ctau_low_mask & mu_mask
            #d0_mu_mask = events.is_ttbar & ctau_mask & chi2_mask & (charm_cand.jetIdx>-1) & np.abs(charm_cand.x_id)==13
            #d0_mu_mask = np.abs(charm_cand.x_id)==13
            musort = ak.argsort(charm_cand.x_pt, ascending=False, axis=-1)
            meson_id = ak.flatten(meson_id)
            meson_id = ak.where(ak.flatten(d0_mu_mask), 42113, meson_id)
            meson_id = ak.unflatten(meson_id, mcount)
            #tmp = ak.cartesian(ak.local_index(ak.fill_none(ak.pad_none(charm_cand.jetIdx[d0_mask]), 1), -1), ak.fill_none(ak.pad_none(charm_cand.jetIdx[d0_mu_mask], 1), -1), nested=True)
            #d0_mu_veto = ak.fill_none(ak.pad_none(~(tmp.slot0 == tmp.slot1), 1), False)

            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint32')

            # Lumi mask (for data)
            #selections.add("is_good_lumi",lumi_mask)

            # ttbar
            #selections.add('ttbar', events.is_ttbar)

            #selections.add('ctau', ak.flatten(ctau_mask))
            #selections.add('vtx', ak.flatten(chi2_mask))

            # J/Psi
            #selections.add('jpsi', ak.flatten(events.is_ttbar & jpsi_mask))
            njpsi = ak.num(events.is_ttbar & jpsi_mask)

            # D0
            #selections.add('d0', ak.flatten(events.is_ttbar & d0_mask & ~ak.any(d0_mu_mask, -1)))
            nd0 = ak.num(events.is_ttbar & d0_mask & ~d0_mu_mask)
            #selections.add('d0_pik', ak.flatten(maskpik))
            #selections.add('d0_kk', ak.flatten(maskkk))
            #selections.add('d0_pipi', ak.flatten(maskpipi))
            #selections.add('d0_unmatched', ak.flatten((~maskpik & ~maskkk & ~maskpipi)))

            # D0mu
            #selections.add('d0mu', ak.flatten(events.is_ttbar & d0_mask & d0_mu_mask))
            nd0mu = ak.any(events.is_ttbar & d0_mask & d0_mu_mask)

            # Counts
            counts = np.ones_like(events['event'])

            # Variables we will loop over when filling hists
            varnames = {}
            jet_flav = ak.ones_like(jets[charm_cand.jetIdx].jetId)
            #jet_flav = ak.ones_like(goodJets[charm_cand.jetIdx].jetId)
            if not isData:
                #jet_flav = goodJets[charm_cand.jetIdx].hadronFlavour
                jet_flav = jets[charm_cand.jetIdx].hadronFlavour
            #xb    = charm_cand.pt / goodJets[charm_cand.jetIdx].pt
            xb    = charm_cand.pt / jets[charm_cand.jetIdx].pt
            xb_ch = charm_cand.fit_pt / charm_cand.j_pt_ch
            xb_ch_g = xb_ch
            if 'gen_d0_pt' in charm_cand:
                #xb_ch_g = charm_cand.gen_d0_pt / goodJets.matched_gen.pt[charm_cand.jetIdx]
                xb_ch_g = charm_cand.gen_d0_pt / jets.matched_gen.pt[charm_cand.jetIdx]
            xb_x    = frag['fragCP5BL_smooth'].values()[0]
            xb_nom  = frag['fragCP5BL_smooth'].values()[1]
            xb_up   = frag['fragCP5BLup_smooth'].values()[1]
            xb_down = frag['fragCP5BLdown_smooth'].values()[1]
            cos2D   = charm_cand.fit_cos2D
            #xb_up   *= np.sum(xb_nom) / np.sum(xb_up)
            #xb_down *= np.sum(xb_nom) / np.sum(xb_down)
            varnames["xb"]    = xb
            varnames["xb_ch"] = xb_ch
            #xb_d0mu = xb + charm_cand.x_pt / goodJets.pt[charm_cand.jetIdx]
            xb_d0mu = xb + charm_cand.x_pt / jets.pt[charm_cand.jetIdx]
            xb_d0mu_ch = xb_ch + charm_cand.x_pt / charm_cand.j_pt_ch
            varnames["xb_d0mu"] = xb_d0mu
            varnames["xb_d0mu_ch"] = xb_d0mu_ch
            varnames["xb_mass_d0"] = (xb_ch, mass, xb_ch_g)
            varnames["xb_mass_d0mu"] = (xb_d0mu_ch, mass, xb_ch_g)
            varnames["xb_mass_jpsi"] = (xb_ch, mass, xb_ch_g)
            varnames["cos2D"] = cos2D
            varnames["nmeson"] = ak.zeros_like(xb_ch)
            varnames["ctau"] = ctau
            varnames["ht"]   = ht
            varnames["met"]  = met.pt
            varnames["j0pt"] = j0pt
            varnames["j_pt_ch"] = ak.fill_none(ak.pad_none(charm_cand.j_pt_ch, 1), 0)
            varnames["njets"]  = nj
            varnames["b0pt"]   = b0pt
            varnames["nbjets"] = nbj
            varnames["l0pt"]   = l0pt
            varnames["nvtx"]   = events.PV.npvs
            varnames["nleps"]  = nl


            ########## Fill the histograms ##########

            # This dictionary keeps track of which selections go with which SR categories
            sr_cat_dict = {
                "ttbar" : {
                    "cuts" : events.is_ttbar & b_mask & pass_trg & cand_mask,# & chi2_mask,
                },
                "d0" : {
                    "cuts" : d0_mask & ~d0_mu_mask & pass_trg,# & chi2_mask,
                    "num" : nd0,
                },
                "d0mu" : {
                    "cuts" : events.is_ttbar & pass_trg & d0_mu_mask,# & ctau_mask & chi2_mask,
                    "num" : nd0mu,
                },
                "jpsi" : {
                    "cuts" : jpsi_mask & pass_trg,# & chi2_mask,
                    "num" : njpsi,
                },
            }

            lep_cat_dict = {
                "mu" : 
                    (ak.num(mu[mu.pt > 26])==1) & \
                    (ak.num(ele) == 0) & \
                    (ak.num(mu[mu.isVetoLep & ~mu.isTightLep]) == 0),
                #"ele" : 
                #    (ak.num(mu) == 0) & (ak.num(ele[ele.pt > 35]) == 1) & (ak.num(ele[ele.pt < 35]) == 0),
                #"mm" : 
                #    (ak.num(mu[mu.pt > 20]) == 2) & (ak.num(ele) == 0) & (events.minMllAFAS > 20) & (met.pt > 20),
                #"ee" : 
                #    (ak.num(mu) == 0) & (ak.num(ele[ele.pt > 30]) == 2) & (events.minMllAFAS > 20) & (met.pt > 20),
                #"em" : 
                #    (ak.num(mu[mu.pt > 20]) == 1) & (ak.num(ele[ele.pt > 30]) == 1) & (events.minMllAFAS > 20),
            }

            # This dictionary keeps track of which selections go with which CR categories
            cr_cat_dict = {
                "qcd" : {
                    "cuts" : events.is_qcd & ~cand_mask & ht_mask & pass_trg,
                    #"cuts" : events.is_qcd & ht_mask & ~b_mask & pass_trg & cand_mask & chi2_mask,
                },
            }

            # Include SRs and CRs unless we asked to skip them
            cat_dict = {}
            if not self._skip_signal_regions:
                cat_dict.update(sr_cat_dict)
            if not self._skip_control_regions:
                cat_dict.update(cr_cat_dict)
            if (not self._skip_signal_regions and not self._skip_control_regions):
                for k in sr_cat_dict:
                    if k in cr_cat_dict:
                        raise Exception(f"The key {k} is in both CR and SR dictionaries.")




            # Set up the list of syst wgt variations to loop over
            wgt_var_lst = ["nominal"]
            if self._do_systematics:
                if not isData:
                    if (syst_var != "nominal"):
                        # In this case, we are dealing with systs that change the kinematics of the objs (e.g. JES)
                        # So we don't want to loop over up/down weight variations here
                        wgt_var_lst = [syst_var]
                    else:
                        # Otherwise we want to loop over the up/down weight variations
                        wgt_var_lst = wgt_var_lst + wgt_correction_syst_lst + data_syst_lst
                else:
                    # This is data, so we want to loop over just up/down variations relevant for data (i.e. FF up and down)
                    wgt_var_lst = wgt_var_lst + data_syst_lst

            # Loop over the sum of weights hists we want to fill
            weights_object = weights_obj_base_for_kinematic_syst
            for wgt_fluct in wgt_var_lst:
                if (wgt_fluct == "nominal") or (wgt_fluct in obj_correction_syst_lst):
                    # In the case of "nominal", or the jet energy systematics, no weight systematic variation is used
                    syst_weight = np.nan_to_num(weights_object.weight(None) / (xsec * lumi), nan=0, posinf=0, neginf=0 )
                    if not isData:
                        hout['sumw'][dataset]  = ak.sum(syst_weight, axis=0)
                        hout['sumw2'][dataset] = ak.sum(np.square(syst_weight), axis=0)
                else:
                    pass
                    #syst_weight = weights_object.weight(wgt_fluct) / (xsec * lumi)
                    #if not isData:
                    #    if f'sumw{wgt_fluct}' not in hout:#self.output:
                    #        hout[f'sumw{wgt_fluct}'] = processor.defaultdict_accumulator(float)
                    #    hout[f'sumw{wgt_fluct}'] = ak.sum(np.nan_to_num(syst_weight, nan=0, posinf=0, neginf=0), axis=0)
                    #    #    self.output[f'sumw{wgt_fluct}'] = processor.defaultdict_accumulator(float)
                    #    #self.output[f'sumw{wgt_fluct}'] = ak.sum(np.nan_to_num(syst_weight, nan=0, posinf=0, neginf=0), axis=0)


            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in varnames.items():
                if dense_axis_name not in self._hist_lst:
                    print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                    continue

                # Loop over the systematics
                for wgt_fluct in wgt_var_lst:

                    # Loop over nlep categories "2l", "3l", "4l"
                    for cat_chan in cat_dict.keys():
                        if dense_axis_name == 'b0pt' and cat_chan == 'qcd': continue
                        # Need to do this inside of nlep cat loop since some wgts depend on lep cat
                        weights_object = weights_dict[cat_chan]

                        if (wgt_fluct == "nominal") or (wgt_fluct in obj_correction_syst_lst):
                            # In the case of "nominal", or the jet energy systematics, no weight systematic variation is used
                            weight = weights_object.weight(None)
                        else:
                            # Otherwise get the weight from the Weights object
                            if wgt_fluct in weights_object.variations:
                                weight = weights_object.weight(wgt_fluct)
                            else:
                                # Note in this case there is no up/down fluct for this cateogry, so we don't want to fill a hist for it
                                continue

                        # This is a check ot make sure we guard against any unintentional variations being applied to data
                        if self._do_systematics and isData:
                            # In all other cases, the up/down variations should correspond to only the ones in the data list
                            if weights_object.variations != set(data_syst_lst): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{set(data_syst_lst)}\" but have \"{weights_object.variations}\".")

                        cuts_lst = cat_dict[cat_chan]['cuts']
                        if 'xb_mass' in dense_axis_name and cat_chan not in dense_axis_name:
                            continue
                        if 'd0mu' in dense_axis_name and 'd0mu' not in cat_chan:
                            continue
                        if 'd0mu' not in dense_axis_name and 'd0mu' in cat_chan:
                            continue
                        #if isData:
                        #    cuts_lst.append("is_good_lumi")
                        #if self._split_by_lepton_flavor:
                        #    flav_ch = lep_flav
                        #    cuts_lst.append(lep_flav)
                        #if dense_axis_name != "njets":
                        #    pass
                        #ch_name = construct_cat_name(lep_chan,njet_str=njet_ch,flav_str=flav_ch)

                        for lep_cut in lep_cat_dict.values():
                            # Get the cuts mask for all selections
                            #all_cuts_mask = cuts_lst#cat_dict[cat_chan]['cuts']
                            all_cuts_mask = cuts_lst & lep_cut#cat_dict[cat_chan]['cuts']

                            # Apply the optional cut on energy of the event
                            if self._ecut_threshold is not None:
                                all_cuts_mask = (all_cuts_mask & ecut_mask)

                            # Weights
                            if cat_chan == 'ttbar' or cat_chan == 'qcd':
                                all_cuts_mask = ak.fill_none(ak.fill_none(ak.pad_none(all_cuts_mask, 1), False), False, axis=0)
                            weights_flat = np.nan_to_num(weight[ak.fill_none(ak.any(all_cuts_mask, -1), False)], nan=0, posinf=0, neginf=0)

                            # Fill the histos
                            sort = ptsort#chi2sort
                            #if 'mu' in dense_axis_name:
                            #    sort = musort
                            if 'xb_mass' in dense_axis_name:
                                xb_val       = ak.flatten(ak.firsts(dense_axis_vals[0][sort][all_cuts_mask], axis=-1), 0)
                                mass_val     = ak.flatten(ak.firsts(dense_axis_vals[1][sort][all_cuts_mask], axis=-1), 0)
                                xb_gen       = copy.deepcopy(ak.flatten(ak.firsts(dense_axis_vals[0][sort][all_cuts_mask], axis=-1), 0))
                                tmp          = ak.pad_none(ak.flatten(ak.firsts(dense_axis_vals[2][sort][all_cuts_mask], axis=-1), 0), len(xb_gen), clip=True, axis=0)
                                # Use gen xb value when possible
                                xb_gen       = ak.where(tmp>0, tmp, xb_gen)
                                meson_id_val = ak.flatten(ak.firsts(meson_id[sort][all_cuts_mask], axis=-1), 0)
                                jet_flav_val = ak.flatten(ak.firsts(jet_flav[sort][all_cuts_mask], axis=-1), 0)
                                #FIXME use gen values
                                xbNom  = np.interp(ak.fill_none(xb_gen, -1), xb_x, xb_nom)
                                xbUp   = np.interp(ak.fill_none(xb_gen, -1), xb_x, xb_up)
                                xbDown = np.interp(ak.fill_none(xb_gen, -1), xb_x, xb_down)
                                axes_fill_info_dict = {
                                    "xb"            : xb_val,
                                    "mass"          : mass_val,
                                    "meson_id"      : meson_id_val,
                                    "jet_flav"      : jet_flav_val,
                                    "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "cut"           : cat_chan,
                                    "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "weight"        : weights_flat,
                                }
                                axes_fill_info_dict_nom = {
                                    "xb"            : xb_val,
                                    "mass"          : mass_val,
                                    "meson_id"      : meson_id_val,
                                    "jet_flav"      : jet_flav_val,
                                    "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "cut"           : cat_chan,
                                    "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "weight"        : weights_flat*xbNom,#[all_cuts_mask],
                                }
                                axes_fill_info_dict_up = {
                                    "xb"            : xb_val,
                                    "mass"          : mass_val,
                                    "meson_id"      : meson_id_val,
                                    "jet_flav"      : jet_flav_val,
                                    "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "cut"           : cat_chan,
                                    "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "weight"        : weights_flat*xbUp,#[all_cuts_mask],
                                }
                                axes_fill_info_dict_down = {
                                    "xb"            : xb_val,
                                    "mass"          : mass_val,
                                    "meson_id"      : meson_id_val,
                                    "jet_flav"      : jet_flav_val,
                                    "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "cut"           : cat_chan,
                                    "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                    "weight"        : weights_flat*xbDown,#[all_cuts_mask],
                                }

                                hout[dense_axis_name].fill(**axes_fill_info_dict)
                                hout[dense_axis_name+'_nom'].fill(**axes_fill_info_dict_nom)
                                hout[dense_axis_name+'_up'].fill(**axes_fill_info_dict_up)
                                hout[dense_axis_name+'_down'].fill(**axes_fill_info_dict_down)
                            else:
                                #all_cuts_mask = ak.any(all_cuts_mask, axis=-1)
                                if dense_axis_name[0] == 'n':
                                    all_cuts_mask = ak.any(all_cuts_mask, axis=-1)
                                    axes_fill_info_dict = {
                                        dense_axis_name : ak.flatten(dense_axis_vals[all_cuts_mask], 0),
                                        "meson_id"      : ak.flatten(ak.firsts(meson_id[all_cuts_mask], axis=-1), 0),
                                        "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                        "cut"           : cat_chan,
                                        "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                        "weight"        : weights_flat,
                                    }
                                elif any(d in dense_axis_name for d in ['ht','met']):
                                    all_cuts_mask = ak.any(all_cuts_mask, axis=-1)
                                    axes_fill_info_dict = {
                                        dense_axis_name : ak.flatten(dense_axis_vals[all_cuts_mask], 0),
                                        "meson_id"      : ak.flatten(ak.firsts(meson_id[all_cuts_mask], axis=-1), 0),
                                        "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                        "cut"           : cat_chan,
                                        "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                        "weight"        : weights_flat,
                                    }
                                else:
                                    if '0pt' in dense_axis_name: all_cuts_mask = ak.any(all_cuts_mask, axis=-1)
                                    axes_fill_info_dict = {
                                        dense_axis_name : ak.flatten(ak.firsts(dense_axis_vals[all_cuts_mask], axis=-1), 0),
                                        "meson_id"      : ak.flatten(ak.firsts(meson_id[all_cuts_mask], axis=-1), 0),
                                        "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                        "cut"           : cat_chan,
                                        "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                                        "weight"        : weights_flat,
                                    }
                                #if '0pt' in dense_axis_name or dense_axis_name[0] == 'n' or 'j_pt_ch' in dense_axis_name:
                                #if 'xb_mass' not in dense_axis_name and 'ctau' not in dense_axis_name:
                                #    axes_fill_info_dict.pop('meson_id')
                                if 'ctau' in dense_axis_name:
                                    if cat_chan == 'qcd':
                                        jet_flav_val = ak.flatten(ak.firsts(ak.ones_like(jets.jetId)[all_cuts_mask], axis=-1), 0)
                                    else:
                                        jet_flav_val = ak.flatten(ak.firsts(jet_flav[sort][all_cuts_mask], axis=-1), 0)
                                    axes_fill_info_dict['jet_flav'] = jet_flav_val
                                if 'nmeson' in dense_axis_name:
                                    axes_fill_info_dict[dense_axis_name] = ak.num(dense_axis_vals[all_cuts_mask])
                                    axes_fill_info_dict[dense_axis_name] = axes_fill_info_dict[dense_axis_name][axes_fill_info_dict[dense_axis_name]>0]
                                    axes_fill_info_dict['weight'] = ak.ones_like(weights_flat)
                                hout[dense_axis_name].fill(**axes_fill_info_dict)

                            # Do not loop over lep flavors if not self._split_by_lepton_flavor, it's a waste of time and also we'd fill the hists too many times
                            #if not self._split_by_lepton_flavor: break

                            # Do not loop over njets if hist is njets (otherwise we'd fill the hist too many times)
                            #if dense_axis_name == "njets": break

        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)
