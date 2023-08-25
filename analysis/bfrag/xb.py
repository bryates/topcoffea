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
        #self.systematics = ['nominal', 'FSRup', 'FSRdown', 'ISRup', 'ISRdown']

        dataset_axis = hist.axis.StrCategory(name="dataset", label="", categories=[], growth=True)
        jpt_axis = hist.axis.Regular(name="j0pt", label="Leading jet $p_{\mathrm{T}}$ [GeV]", bins=50, start=0, stop=300)
        bpt_axis = hist.axis.Regular(name="b0pt", label="Leading b jet $p_{\mathrm{T}}$ [GeV]", bins=50, start=0, stop=300)
        lpt_axis = hist.axis.Regular(name="l0pt", label="Leading lepotn $p_{\mathrm{T}}$ [GeV]", bins=50, start=0, stop=100)
        D0pt_axis= hist.axis.Regular(name="D0pt", label="Leading D0 $p_{\mathrm{T}}$ [GeV]", bins=50, start=0, stop=100)
        D0pipt_axis= hist.axis.Regular(name="D0pipt", label="Leading D0 pi $p_{\mathrm{T}}$ [GeV]", bins=50, start=0, stop=100)
        D0kpt_axis= hist.axis.Regular(name="D0kpt", label="Leading D0 k $p_{\mathrm{T}}$ [GeV]", bins=50, start=0, stop=100)
        xb_jpsi_axis  = hist.axis.Regular(name="xb_jpsi",   label="$x_{\mathrm{b}}$", bins=10, start=0, stop=1)
        xb_axis  = hist.axis.Regular(name="xb",   label="$x_{\mathrm{b}}$", bins=10, start=0, stop=1)
        xb_ch_axis  = hist.axis.Regular(name="xb_ch",   label="$x_{\mathrm{b}} \Sigma p_{\mathrm{T}}^{\mathrm{charged}}$", bins=10, start=0, stop=1)
        HT_axis  = hist.axis.Regular(name="HT",   label="$H_{\mathrm{T}}$", bins=50, start=0, stop=1000)
        pdgid_axis= hist.axis.Regular(name="pdgid",   label="D0 id's", bins=10, start=0, stop=250)
        d0_axis  = hist.axis.Regular(name='d0',   label="$d_0$", bins=10, start=0, stop=100)
        njets_axis = hist.axis.Regular(name='njets', label='$N_{\mathrm{jets}}$', bins=10, start=0, stop=10)
        nbjets_axis = hist.axis.Regular(name='nbjets', label='$N_{\mathrm{b-jets}}$', bins=10, start=0, stop=10)
        nleps_axis = hist.axis.Regular(name='nleps', label='$N_{\mathrm{leps}}$', bins=10, start=0, stop=10)
        jpsi_mass_axis = hist.axis.Regular(name='mass', label='J/Psi mass [GeV]', bins=len(self.jpsi_mass_bins), start=self.jpsi_mass_bins[0], stop=self.jpsi_mass_bins[-1])
        d0_mass_axis = hist.axis.Regular(name='mass', label='D0 mass [GeV]', bins=len(self.d0_mass_bins), start=self.d0_mass_bins[0], stop=self.d0_mass_bins[-1])
        mass_axes = [hist.axis.Regular(name=f'd0_{int(xb_bin*10)}', label='D0 mass [GeV] (' + str(round(self.xb_bins[ibin], 2)) + ' < $x_{\mathrm{b}}$ < ' + str(round(self.xb_bins[ibin+1], 2)) + ')', bins=len(self.d0_mass_bins), start=self.d0_mass_bins[0], stop=self.d0_mass_bins[-1]) for ibin,xb_bin in enumerate(self.xb_bins[:-1])]
        meson_axis = hist.axis.IntCategory(name="meson_id", label="Meson pdgId (421 D0, 42113 D0mu, 443 J/Psi, 443211 J/Psi+K)", categories=[421, 42113, 443])
        jet_axis = hist.axis.IntCategory(name="jet_id", label="Jet flavor", categories=list(range(1,7)))
        pi_gen_axis = hist.axis.IntCategory(name="pi_gid", label="Gen-matched flavor", categories=[], growth=True)
        k_gen_axis = hist.axis.IntCategory(name="k_gid", label="Gen-matched flavor", categories=[], growth=True)
        pi_mother_gen_axis = hist.axis.IntCategory(name="pi_mother", label="Gen-matched flavor", categories=[], growth=True)
        k_mother_gen_axis = hist.axis.IntCategory(name="k_mother", label="Gen-matched flavor", categories=[], growth=True)
        ctau_axis = hist.axis.Regular(name='ctau', label='Meson time of flight', bins=100, start=0, stop=100)
        systematic_axis = hist.axis.StrCategory(name='systematic', categories=['nominal'],growth=True)
        self.output = processor.dict_accumulator({
            'j0pt': hist.Hist(dataset_axis, jpt_axis, systematic_axis),
            'b0pt': hist.Hist(dataset_axis, bpt_axis, systematic_axis),
            'l0pt': hist.Hist(dataset_axis, lpt_axis, systematic_axis),
            'D0pt': hist.Hist(dataset_axis, D0pt_axis, systematic_axis),
            'D0pipt': hist.Hist(dataset_axis, D0pipt_axis, systematic_axis),
            'D0kpt': hist.Hist(dataset_axis, D0kpt_axis, systematic_axis),
            'jet_id'  : hist.Hist(dataset_axis, meson_axis, jet_axis, systematic_axis),
            'xb_mass_jpsi'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, systematic_axis),
            'xb_mass_d0mu'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_jpsi_up'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, systematic_axis),
            'xb_mass_d0mu_up'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_up'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_jpsi_down'  : hist.Hist(dataset_axis, meson_axis, xb_axis, jpsi_mass_axis, systematic_axis),
            'xb_mass_d0mu_down'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_down'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_pik'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_kk'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_pipi'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_unmatched'  : hist.Hist(dataset_axis, meson_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_mass_d0_gen'  : hist.Hist(dataset_axis, meson_axis, pi_gen_axis, k_gen_axis, pi_mother_gen_axis, k_mother_gen_axis, xb_axis, d0_mass_axis, systematic_axis),
            'xb_jpsi'  : hist.Hist(dataset_axis, xb_jpsi_axis, systematic_axis),
            'xb_d0'  : hist.Hist(dataset_axis, xb_axis, systematic_axis),
            'xb_d0mu'  : hist.Hist(dataset_axis, xb_axis, systematic_axis),
            'xb_ch'  : hist.Hist(dataset_axis, xb_ch_axis, systematic_axis),
            'HT'  : hist.Hist(dataset_axis, HT_axis, systematic_axis),
            'pdgid'  : hist.Hist(dataset_axis, pdgid_axis, systematic_axis),
            'd0'  : hist.Hist(dataset_axis, d0_axis, systematic_axis),
            'njets' : hist.Hist(dataset_axis, njets_axis, systematic_axis),
            'nbjets' : hist.Hist(dataset_axis, nbjets_axis, systematic_axis),
            'nleps' : hist.Hist(dataset_axis, nleps_axis, systematic_axis),
            'jpsi_mass': hist.Hist(dataset_axis, meson_axis, jpsi_mass_axis, systematic_axis),
            'd0_mass': hist.Hist(dataset_axis, meson_axis, d0_mass_axis, systematic_axis),
            'ctau': hist.Hist(dataset_axis, ctau_axis, meson_axis, systematic_axis),
            'vtx_mass_d0' : hist.Hist(dataset_axis, hist.axis.Regular(name='vtx', label='Vertex prob.', bins=100, start=0, stop=.1), d0_mass_axis, systematic_axis),
            'chi_mass_d0' : hist.Hist(dataset_axis, hist.axis.Regular(name='chi', label='$\chi^2$ vtx', bins=100, start=0, stop=5), d0_mass_axis, systematic_axis),
            'vtx_mass_jpsi' : hist.Hist(dataset_axis, hist.axis.Regular(name='vtx', label='Vertex prob.', bins=100, start=0, stop=.1), jpsi_mass_axis, systematic_axis),
            'chi_mass_jpsi' : hist.Hist(dataset_axis, hist.axis.Regular(name='chi', label='$\chi^2$ vtx', bins=100, start=0, stop=5), jpsi_mass_axis, systematic_axis),
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
        frag = uproot.open(topcoffea_path('analysis/bfrag/bfragweights.root'))

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
        for d in datasets:
            if d in dataset: dataset = dataset.split('_')[0]

        # Set the sampleType (used for MC matching requirement)
        sampleType = "prompt"
        if isData:
            sampleType = "data"

        # Initialize objects
        met  = events.MET
        ele  = events.Electron
        mu   = events.Muon
        jets = events.Jet

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(events.MET.pt)

        leptons = ak.with_name(ak.concatenate([ele, mu], axis=1), 'PtEtaPhiMCandidate')
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
        j0pt = ak.fill_none(ak.firsts(jets.pt), -1)
        l0pt = ak.fill_none(ak.firsts(leptons.pt), -1)

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
            "PUUp","PUDown", # Exp systs
            #f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown", # Exp systs
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
            l_fo = ak.with_name(ak.concatenate([ele, mu], axis=1), 'PtEtaPhiMCandidate')
            vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
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
            cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
            goodJets = jets#cleanedJets[cleanedJets.isGood]

            # Count jets
            njets = ak.num(goodJets)
            ht = ak.sum(goodJets.pt,axis=-1)
            j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]

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
            # Move to selection.py
            def is_ttbar(jets, bjets, leptons):
                jets_pt = ak.fill_none(ak.pad_none(jets.pt, 1), 0)
                leps_pt  = ak.fill_none(ak.pad_none(leptons.pt, 1), 0)
                
                jets_eta = ak.fill_none(ak.pad_none(jets.eta, 1), 0)
                leps_eta  = ak.fill_none(ak.pad_none(leptons.eta, 1), 0)
                
                nJets = ak.num(jets)
                nBJets = ak.num(bjets)
                nLep = ak.num(leptons)
                jpt30 = jets_pt[:,0] > 30
                lpt25 = (leps_pt[:,0] > 25)
                jeta24 = np.abs(jets_eta) < 2.4
                leta24 = np.abs(leps_eta) < 2.4
                #is_ttbar = (ak.num(jets)>4)&(ak.num(bjets_tight)>1)&((ak.num(ele)>1)|(ak.num(mu)>1))# & (pad_jets[:,0].pt>30)# & ((pad_ele[:,0].pt>25) | (pad_mu[:,0].pt>25))#&(ht>180)
                return (nJets >= 1) & (nBJets >=1) & (nLep >= 1) & jpt30 & lpt25# & jeta24 & leta24

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
            isBtagJetsTight = (jets.btagDeepFlavB > btagwpt)
            isNotBtagJetsTight = np.invert(isBtagJetsTight)
            nbtagst = ak.num(jets[isBtagJetsTight])

            bjets_tight = ak.pad_none(jets[isBtagJetsTight], 1)
            events['is_ttbar'] = is_ttbar(jets, bjets_tight, leptons)


            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets

            ######### Event weights that do not depend on the lep cat ##########

            if not isData:

                # Btag SF following 1a) in https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
                isBtagJetsLooseNotMedium = (isBtagJetsLoose & isNotBtagJetsMedium)
                bJetSF   = [GetBTagSF(goodJets, year, 'LOOSE'),GetBTagSF(goodJets, year, 'MEDIUM')]
                bJetEff  = [GetBtagEff(goodJets, year, 'loose'),GetBtagEff(goodJets, year, 'medium')]
                bJetEff_data   = [bJetEff[0]*bJetSF[0],bJetEff[1]*bJetSF[1]]
                pMC     = ak.prod(bJetEff[1]       [isBtagJetsMedium], axis=-1) * ak.prod((bJetEff[0]       [isBtagJetsLooseNotMedium] - bJetEff[1]       [isBtagJetsLooseNotMedium]), axis=-1) * ak.prod((1-bJetEff[0]       [isNotBtagJetsLoose]), axis=-1)
                pMC     = ak.where(pMC==0,1,pMC) # removeing zeroes from denominator...
                pData   = ak.prod(bJetEff_data[1]  [isBtagJetsMedium], axis=-1) * ak.prod((bJetEff_data[0]  [isBtagJetsLooseNotMedium] - bJetEff_data[1]  [isBtagJetsLooseNotMedium]), axis=-1) * ak.prod((1-bJetEff_data[0]  [isNotBtagJetsLoose]), axis=-1)
                weights_obj_base_for_kinematic_syst.add("btagSF", pData/pMC)

                if self._do_systematics and syst_var=='nominal':
                    for b_syst in ["bc_corr","light_corr",f"bc_{year}",f"light_{year}"]:
                        bJetSFUp = [GetBTagSF(goodJets, year, 'LOOSE', syst=b_syst)[0],GetBTagSF(goodJets, year, 'MEDIUM', syst=b_syst)[0]]
                        bJetSFDo = [GetBTagSF(goodJets, year, 'LOOSE', syst=b_syst)[1],GetBTagSF(goodJets, year, 'MEDIUM', syst=b_syst)[1]]
                        bJetEff_dataUp = [bJetEff[0]*bJetSFUp[0],bJetEff[1]*bJetSFUp[1]]
                        bJetEff_dataDo = [bJetEff[0]*bJetSFDo[0],bJetEff[1]*bJetSFDo[1]]
                        pDataUp = ak.prod(bJetEff_dataUp[1][isBtagJetsMedium], axis=-1) * ak.prod((bJetEff_dataUp[0][isBtagJetsLooseNotMedium] - bJetEff_dataUp[1][isBtagJetsLooseNotMedium]), axis=-1) * ak.prod((1-bJetEff_dataUp[0][isNotBtagJetsLoose]), axis=-1)
                        pDataDo = ak.prod(bJetEff_dataDo[1][isBtagJetsMedium], axis=-1) * ak.prod((bJetEff_dataDo[0][isBtagJetsLooseNotMedium] - bJetEff_dataDo[1][isBtagJetsLooseNotMedium]), axis=-1) * ak.prod((1-bJetEff_dataDo[0][isNotBtagJetsLoose]), axis=-1)
                        weights_obj_base_for_kinematic_syst.add(f"btagSF{b_syst}", events.nom, (pDataUp/pMC)/(pData/pMC),(pDataDo/pMC)/(pData/pMC))

                # Trigger SFs
                #GetTriggerSF(year,events,leptons[:,0],leptons[:,1])
                #weights_obj_base_for_kinematic_syst.add(f"triggerSF_{year}", events.trigger_sf, copy.deepcopy(events.trigger_sfUp), copy.deepcopy(events.trigger_sfDown))            # In principle does not have to be in the lep cat loop


            ######### Event weights that do depend on the lep cat ###########

            # Loop over categories and fill the dict
            weights_dict = {}
            for ch_name in ["d0", "d0mu", "jpsi"]:

                # For both data and MC
                weights_dict[ch_name] = copy.deepcopy(weights_obj_base_for_kinematic_syst)

                # For MC only
                '''
                if not isData:
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

            # b jet masks
            bmask_atleast1tight = (nbtagst>=1) # Used for tttt enriched

            # Charm meson candidates from b-jets
            charm_cand = events.BToCharm
            ptsort = ak.argsort(charm_cand.pt, ascending=False)
            charm_cand = events.BToCharm[ptsort]
            ctau_mask = ak.fill_none((charm_cand.vtx_l3d / charm_cand.vtx_el3d) > 10, False)
            ctau = ak.firsts((charm_cand.vtx_l3d / charm_cand.vtx_el3d))
            chi2_mask = (charm_cand.svprob>0.02) & (charm_cand.chi2<5)
            ht_mask = ht > 180
            b_mask = jets[charm_cand.jetIdx].btagDeepFlavB > btagwpt
            d0_mask = chi2_mask & b_mask & (charm_cand.jetIdx>-1) & (np.abs(charm_cand.meson_id) == 421)
            jpsi_mask = chi2_mask & b_mask & (charm_cand.jetIdx>-1) & (np.abs(charm_cand.meson_id) == 443)
            d0_mask = ht_mask & chi2_mask & (charm_cand.jetIdx>-1) & (np.abs(charm_cand.meson_id) == 421)
            jpsi_mask = chi2_mask & (charm_cand.jetIdx>-1) & (np.abs(charm_cand.meson_id) == 443)
            mass = ak.firsts(charm_cand.fit_mass)

            # D0 mu tagged
            pi_gid = ak.firsts(ak.fill_none(charm_cand.pigId, 0))
            k_gid = ak.firsts(ak.fill_none(charm_cand.kgId, 0))
            pi_mother = ak.firsts(ak.fill_none(charm_cand.pi_mother, 0))
            k_mother = ak.firsts(ak.fill_none(charm_cand.k_mother, 0))
            #d0_gid = np.abs(pi_gid)*1e6 + np.abs(k_gid)*1e3 + np.abs(
            maskpik = ((abs(pi_gid)==211) & (abs(k_gid)==321))
            maskkk = ((abs(pi_gid)==321) & (abs(k_gid)==321))
            maskpipi = ((abs(pi_gid)==211) & (abs(k_gid)==211))
 
            # D0 mu tagged
            d0_mu_mask = ak.any(np.abs(charm_cand.x_id)==13, -1)
            charm_cand[d0_mu_mask]['meson_id'] = 42113
            meson_id = ak.firsts(charm_cand.meson_id)

            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint32')

            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)

            selections.add('ctau', ak.firsts(ctau_mask, axis=1))
            selections.add('vtx', ak.firsts(chi2_mask, axis=1))

            # ttbar
            selections.add('ttbar', events.is_ttbar)

            # J/Psi
            selections.add('jpsi', ak.firsts(jpsi_mask, axis=1))

            # D0
            selections.add('d0', ak.firsts(d0_mask & ~d0_mu_mask, axis=-1))
            selections.add('d0_pik', maskpik)
            selections.add('d0_kk', maskkk)
            selections.add('d0_pipi', maskpipi)
            selections.add('d0_unmatched', (~maskpik & ~maskkk & ~maskpipi))

            # D0mu
            selections.add('d0mu', ak.firsts(d0_mask & d0_mu_mask, axis=-1))

            # Counts
            counts = np.ones_like(events['event'])

            # Variables we will loop over when filling hists
            varnames = {}
            xb    = ak.firsts(charm_cand.pt / jets[charm_cand.jetIdx].pt)
            xb_ch = ak.firsts(charm_cand.fit_pt / charm_cand.j_pt_ch)
            xb_x    = frag['fragCP5BL_smooth'].values()[0]
            xb_nom  = frag['fragCP5BL_smooth'].values()[1]
            xb_up   = frag['fragCP5BLup_smooth'].values()[1]
            xb_down = frag['fragCP5BLdown_smooth'].values()[1]
            #xb_up   *= np.sum(xb_nom) / np.sum(xb_up)
            #xb_down *= np.sum(xb_nom) / np.sum(xb_down)
            #FIXME use gen values
            xbNom   = np.interp(ak.fill_none(xb, -1), xb_x, xb_nom)
            xbUp   = np.interp(ak.fill_none(xb, -1), xb_x, xb_up)
            xbDown = np.interp(ak.fill_none(xb, -1), xb_x, xb_down)
            varnames["xb"]    = xb
            varnames["xb_ch"] = xb_ch
            xb_d0mu = ak.firsts(xb + charm_cand.x_pt / jets.pt[charm_cand.jetIdx])
            xb_d0mu_ch = ak.firsts(xb + charm_cand.x_pt / charm_cand.j_pt_ch)
            varnames["xb_d0mu"] = xb_d0mu
            varnames["xb_d0mu_ch"] = xb_d0mu_ch
            varnames["xb_mass_d0"] = (xb_ch, mass)
            varnames["xb_mass_d0mu"] = (xb_d0mu_ch, mass)
            varnames["xb_mass_jpsi"] = (xb_ch, mass)


            ########## Fill the histograms ##########

            # This dictionary keeps track of which selections go with which SR categories
            sr_cat_dict = {
                "d0" : {
                    "chan_lst" : ["d0", "ttbar", "ctau", "vtx"],
                },
                "d0mu" : {
                    "chan_lst" : ["d0mu", "ttbar", "ctau", "vtx"],
                },
                "jpsi" : {
                    "chan_lst" : ["jpsi", "ttbar", "ctau", "vtx"],
                },
            }

            # This dictionary keeps track of which selections go with which CR categories
            cr_cat_dict = {
                '''
                "2l_CRflip" : {
                    "atmost_3j" : {
                        "lep_chan_lst" : ["2lss_CRflip"],
                        "lep_flav_lst" : ["ee"],
                        "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
                    },
                },
                "2l_CR" : {
                    "exactly_1j" : {
                        "lep_chan_lst" : ["2lss_CR"],
                        "lep_flav_lst" : ["ee" , "em" , "mm"],
                        "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
                    },
                    "exactly_2j" : {
                        "lep_chan_lst" : ["2lss_CR"],
                        "lep_flav_lst" : ["ee" , "em" , "mm"],
                        "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
                    },
                    "exactly_3j" : {
                        "lep_chan_lst" : ["2lss_CR"],
                        "lep_flav_lst" : ["ee" , "em" , "mm"],
                        "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
                    },
                },
                "3l_CR" : {
                    "exactly_0j" : {
                        "lep_chan_lst" : ["3l_CR"],
                        "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                        "appl_lst"     : ["isSR_3l" , "isAR_3l"],
                    },
                    "atleast_1j" : {
                        "lep_chan_lst" : ["3l_CR"],
                        "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                        "appl_lst"     : ["isSR_3l" , "isAR_3l"],
                    },
                },
                "2los_CRtt" : {
                    "exactly_2j"   : {
                        "lep_chan_lst" : ["2los_CRtt"],
                        "lep_flav_lst" : ["em"],
                        "appl_lst"     : ["isSR_2lOS" , "isAR_2lOS"],
                    },
                },
                "2los_CRZ" : {
                    "atleast_0j"   : {
                        "lep_chan_lst" : ["2los_CRZ"],
                        "lep_flav_lst" : ["ee", "mm"],
                        "appl_lst"     : ["isSR_2lOS" , "isAR_2lOS"],
                    },
                },
                '''
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
                    hout['sumw'][dataset]  = ak.sum(syst_weight, axis=0)
                    hout['sumw2'][dataset] = ak.sum(np.square(syst_weight), axis=0)
                else:
                    syst_weight = weights_object.weight(wgt_fluct) / (xsec * lumi)
                    if f'sumw{wgt_fluct}' not in self.output:
                        self.output[f'sumw{wgt_fluct}'] = processor.defaultdict_accumulator(float)
                    self.output[f'sumw{wgt_fluct}'] = ak.sum(np.nan_to_num(syst_weight, nan=0, posinf=0, neginf=0), axis=0)

            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in varnames.items():
                if dense_axis_name not in self._hist_lst:
                    print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                    continue

                # Loop over the systematics
                for wgt_fluct in wgt_var_lst:

                    # Loop over nlep categories "2l", "3l", "4l"
                    for cat_chan in cat_dict.keys():
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
                            # Should not have any up/down variations for data in 4l (since we don't estimate the fake rate there)
                            if nlep_cat == "4l":
                                if weights_object.variations != set([]): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{[]}\" but have \"{weights_object.variations}\".")
                            # In all other cases, the up/down variations should correspond to only the ones in the data list
                            else:
                                if weights_object.variations != set(data_syst_lst): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{set(data_syst_lst)}\" but have \"{weights_object.variations}\".")

                        cuts_lst = cat_dict[cat_chan]['chan_lst']
                        if 'xb_mass' in dense_axis_name and cat_chan not in dense_axis_name:
                            continue
                        if 'd0mu' in dense_axis_name and 'd0mu' not in cat_chan:
                            continue
                        if 'd0mu' not in dense_axis_name and 'd0mu' in cat_chan:
                            continue
                        if isData:
                            cuts_lst.append("is_good_lumi")
                        if self._split_by_lepton_flavor:
                            flav_ch = lep_flav
                            cuts_lst.append(lep_flav)
                        if dense_axis_name != "njets":
                            pass
                        #ch_name = construct_cat_name(lep_chan,njet_str=njet_ch,flav_str=flav_ch)

                        # Get the cuts mask for all selections
                        if dense_axis_name == "njets":
                            all_cuts_mask = (selections.all(*cuts_lst) & njets_any_mask)
                        else:
                            all_cuts_mask = selections.all(*cuts_lst)

                        # Apply the optional cut on energy of the event
                        if self._ecut_threshold is not None:
                            all_cuts_mask = (all_cuts_mask & ecut_mask)

                        # Weights
                        weights_flat = np.nan_to_num(weight[all_cuts_mask], nan=0, posinf=0, neginf=0)


                        # Fill the histos
                        axes_fill_info_dict = {
                            "xb"            : dense_axis_vals[0][all_cuts_mask],
                            "mass"          : dense_axis_vals[1][all_cuts_mask],
                            "meson_id"      : meson_id[all_cuts_mask],
                            "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                            "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                            "weight"        : weights_flat*xbNom[all_cuts_mask],
                        }
                        axes_fill_info_dict_up = {
                            "xb"            : dense_axis_vals[0][all_cuts_mask],
                            "mass"          : dense_axis_vals[1][all_cuts_mask],
                            "meson_id"      : meson_id[all_cuts_mask],
                            "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                            "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                            "weight"        : weights_flat*xbUp[all_cuts_mask], 
                        }
                        axes_fill_info_dict_down = {
                            "xb"            : dense_axis_vals[0][all_cuts_mask],
                            "mass"          : dense_axis_vals[1][all_cuts_mask],
                            "meson_id"      : meson_id[all_cuts_mask],
                            "dataset"       : dataset, #ak.Array([dataset] * ak.num(meson_id[all_cuts_mask], axis=0)),
                            "systematic"    : wgt_fluct, #ak.Array([wgt_fluct] * ak.num(meson_id[all_cuts_mask], axis=0)),
                            "weight"        : weights_flat*xbDown[all_cuts_mask],
                        }

                        hout[dense_axis_name].fill(**axes_fill_info_dict)
                        hout[dense_axis_name+'_up'].fill(**axes_fill_info_dict_up)
                        hout[dense_axis_name+'_down'].fill(**axes_fill_info_dict_down)

                        # Do not loop over lep flavors if not self._split_by_lepton_flavor, it's a waste of time and also we'd fill the hists too many times
                        if not self._split_by_lepton_flavor: break

                        # Do not loop over njets if hist is njets (otherwise we'd fill the hist too many times)
                        if dense_axis_name == "njets": break

        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)
