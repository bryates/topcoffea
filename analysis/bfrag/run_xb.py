#!/usr/bin/env python

import argparse
import json
import time
import cloudpickle
import gzip
import os

import numpy as np
import hist
import coffea
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.util import save

from xb import AnalysisProcessor
import topcoffea.modules.utils as utils
from topcoffea.modules.dataDrivenEstimation import DataDrivenProducer
from topcoffea.modules.get_renormfact_envelope import get_renormfact_envelope
import topcoffea.modules.remote_environment as remote_environment

LST_OF_KNOWN_EXECUTORS = ["futures","work_queue","dask"]

WGT_VAR_LST = [
    "nSumOfWeights_ISRUp",
    "nSumOfWeights_ISRDown",
    "nSumOfWeights_FSRUp",
    "nSumOfWeights_FSRDown",
    "nSumOfWeights_renormUp",
    "nSumOfWeights_renormDown",
    "nSumOfWeights_factUp",
    "nSumOfWeights_factDown",
    "nSumOfWeights_renormfactUp",
    "nSumOfWeights_renormfactDown",
]

def get_xsec(proc):
    xsec_json = "xsecs.json"
    with open(xsec_json) as f_xsec:
        xsec = json.load(f_xsec)
        xsec = xsec[proc]
    return xsec

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--executor','-x'  , default='work_queue', help = 'Which executor to use')
    parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--test','-t'       , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
    parser.add_argument('--pretend'        , action='store_true', help = 'Read json files but, not execute the analysis')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s' , default=250000, help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None, help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'   , default='coffea_dask', help = 'Name of the output file with histograms')
    parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
    parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--do-errors'      , action='store_true', help = 'Save the w**2 coefficients')
    parser.add_argument('--do-systs', action='store_true', help = 'Compute systematic variations')
    parser.add_argument('--split-lep-flavor', action='store_true', help = 'Split up categories by lepton flavor')
    parser.add_argument('--skip-sr', action='store_true', help = 'Skip all signal region categories')
    parser.add_argument('--skip-cr', action='store_true', help = 'Skip all control region categories')
    parser.add_argument('--do-renormfact-envelope', action='store_true', help = 'Perform renorm/fact envelope calculation on the output hist (saves the modified with the the same name as the original.')
    parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
    parser.add_argument('--hist-list', action='extend', nargs='+', help = 'Specify a list of histograms to fill.')
    parser.add_argument('--ecut', default=None  , help = 'Energy cut threshold i.e. throw out events above this (GeV)')
    parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')
    parser.add_argument('--scaleout', type=int, default=6 , help = 'File to run')
    parser.add_argument('--njobs', type=int, default=4, help='Nuber of jobs')


    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    prefix     = args.prefix
    executor   = args.executor
    dotest     = args.test
    nworkers   = int(args.nworkers)
    chunksize  = int(args.chunksize)
    nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname    = args.outname
    outpath    = args.outpath
    pretend    = args.pretend
    treename   = args.treename
    do_errors  = args.do_errors
    do_systs   = args.do_systs
    split_lep_flavor = args.split_lep_flavor
    skip_sr    = args.skip_sr
    skip_cr    = args.skip_cr
    do_renormfact_envelope = args.do_renormfact_envelope

    # Check if we have valid options
    if executor not in LST_OF_KNOWN_EXECUTORS:
        raise Exception(f"The \"{executor}\" executor is not known. Please specify an executor from the known executors ({LST_OF_KNOWN_EXECUTORS}). Exiting.")
    if do_renormfact_envelope:
        if not do_systs:
            raise Exception("Error: Cannot specify do_renormfact_envelope if we are not including systematics.")
    if dotest:
        if executor == "futures":
            nchunks = 2
            chunksize = 10000
            nworkers = 1
            print('Running a fast test with %i workers, %i chunks of %i events'%(nworkers, nchunks, chunksize))
        else:
            raise Exception(f"The \"test\" option is not set up to work with the {executor} executor. Exiting.")


    # Set the threshold for the ecut (if not applying a cut, should be None)
    ecut_threshold = args.ecut
    if ecut_threshold is not None: ecut_threshold = float(args.ecut)

    if executor == "work_queue":
        # construct wq port range
        port = list(map(int, args.port.split('-')))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert single values into a range of one element
            port.append(port[0])

    # Figure out which hists to include
    if args.hist_list == ["ana"]:
        # Here we hardcode a list of hists used for the analysis
        hist_lst = ['xb_mass_jpsi', 'xb_mass_d0', 'xb_mass_d0mu']
    elif args.hist_list == ["cr"]:
        # Here we hardcode a list of hists used for the CRs
        hist_lst = ["lj0pt", "ptz", "met", "ljptsum", "l0pt", "l0eta", "l1pt", "l1eta", "j0pt", "j0eta", "njets", "nbtagsl", "invmass"]
    else:
        # We want to specify a custom list
        # If we don't specify this argument, it will be None, and the processor will fill all hists
        hist_lst = args.hist_list


    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        fileName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
        if fileName.endswith('.json'): fileName = fileName[:-5]
        with open(jsonFile) as jf:
            tjson = json.load(jf)
            for sampleName in tjson:
                samplesdict[sampleName] = {'files': tjson[sampleName]}
                samplesdict[sampleName]['redirector'] = prefix

    if isinstance(jsonFiles, str) and ',' in jsonFiles:
        jsonFiles = jsonFiles.replace(' ', '').split(',')
    elif isinstance(jsonFiles, str):
        jsonFiles = [jsonFiles]
    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith('/'): jsonFile+='/'
            for f in os.path.listdir(jsonFile):
                if f.endswith('.json'): allInputFiles.append(jsonFile+f)
        else:
            allInputFiles.append(jsonFile)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            raise Exception(f'[ERROR] Input file {f} not found!')
        # This input file is a json file, not a cfg
        if f.endswith('.json'):
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(' >> Reading json from cfg file...')
                lines = fin.readlines()
                for l in lines:
                    if '#' in l:
                        l=l[:l.find('#')]
                    l = l.replace(' ', '').replace('\n', '')
                    if l == '': continue
                    print(l)
                    if ',' in l:
                        l = l.split(',')
                        for nl in l:
                            if not os.path.isfile(l):
                                prefix = nl
                            else:
                                LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l):
                            prefix = l
                        else:
                            LoadJsonToSampleName(l, prefix)

    flist = {}
    nevts_total = 0
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        year = sname.split('_')[-1].replace('20', '')
        if len(year) > 2 and 'APV' not in year:
            year = year[:2]
        year = '20'+year
        samplesdict[sname]['year'] = year
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['xsec'] = get_xsec('_'.join(sname.split('_')[:-1]))
        samplesdict[sname]['isData'] = 'Data' in sname
        # Print file info
        print('>> '+sname)
        print('   - isData?      : %s'   %('YES' if samplesdict[sname]['isData'] else 'NO'))
        print('   - year         : %s'   %samplesdict[sname]['year'])
        print('   - xsec         : %f'   %samplesdict[sname]['xsec'])

    if pretend:
        print('pretending...')
        exit()

    processor_instance = AnalysisProcessor(samplesdict,hist_lst,ecut_threshold,do_errors,do_systs,split_lep_flavor,skip_sr,skip_cr)

    if executor == "work_queue":
        executor_args = {
            'master_name': '{}-workqueue-coffea'.format(os.environ['USER']),

            # find a port to run work queue in this range:
            'port': port,

            'debug_log': 'debug.log',
            'transactions_log': 'tr.log',
            'stats_log': 'stats.log',
            'tasks_accum_log': 'tasks.log',

            'environment_file': remote_environment.get_environment(),
            'extra_input_files': ["xb.py"],

            'retries': 5,

            # use mid-range compression for chunks results. 9 is the default for work
            # queue in coffea. Valid values are 0 (minimum compression, less memory
            # usage) to 16 (maximum compression, more memory usage).
            'compression': 9,

            # automatically find an adequate resource allocation for tasks.
            # tasks are first tried using the maximum resources seen of previously ran
            # tasks. on resource exhaustion, they are retried with the maximum resource
            # values, if specified below. if a maximum is not specified, the task waits
            # forever until a larger worker connects.
            'resource_monitor': True,
            'resources_mode': 'auto',

            # this resource values may be omitted when using
            # resources_mode: 'auto', but they do make the initial portion
            # of a workflow run a little bit faster.
            # Rather than using whole workers in the exploratory mode of
            # resources_mode: auto, tasks are forever limited to a maximum
            # of 8GB of mem and disk.
            #
            # NOTE: The very first tasks in the exploratory
            # mode will use the values specified here, so workers need to be at least
            # this large. If left unspecified, tasks will use whole workers in the
            # exploratory mode.
            # 'cores': 1,
            # 'disk': 8000,   #MB
            # 'memory': 10000, #MB

            # control the size of accumulation tasks. Results are
            # accumulated in groups of size chunks_per_accum, keeping at
            # most chunks_per_accum at the same time in memory per task.
            'chunks_per_accum': 25,
            'chunks_accum_in_mem': 2,

            # terminate workers on which tasks have been running longer than average.
            # This is useful for temporary conditions on worker nodes where a task will
            # be finish faster is ran in another worker.
            # the time limit is computed by multipliying the average runtime of tasks
            # by the value of 'fast_terminate_workers'.  Since some tasks can be
            # legitimately slow, no task can trigger the termination of workers twice.
            #
            # warning: small values (e.g. close to 1) may cause the workflow to misbehave,
            # as most tasks will be terminated.
            #
            # Less than 1 disables it.
            'fast_terminate_workers': 0,

            # print messages when tasks are submitted, finished, etc.,
            # together with their resource allocation and usage. If a task
            # fails, its standard output is also printed, so we can turn
            # off print_stdout for all tasks.
            'verbose': True,
            'print_stdout': False,
        }

    # Run the processor and get the output
    tstart = time.time()

    if executor == "futures":
        exec_instance = processor.FuturesExecutor(workers=nworkers)
        runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)
        output = runner(flist, treename, processor_instance)
    elif executor ==  "dask":
        from distributed import Client
        from dask_jobqueue import HTCondorCluster
        import socket
        def hname():
            import socket
            return socket.gethostname()
        n_port = 8786
        with HTCondorCluster(
                cores=1,
                #memory='10000MB',
                memory='8GB',
                disk='1000MB',
                death_timeout = '300',
                #lcg = True,
                nanny = False,
                #container_runtime = "none",
                #container_runtime = "singularity",
                log_directory = "logs",
                #log_directory = "/afs/cern.ch/user/b/byates/CMSSW_10_6_18/src/BFrag/BFrag/logs",
                #log_directory = "/eos/user/b/byates/condor/log",
                scheduler_options={
                    'port': n_port,
                    'host': socket.gethostname(),
                    },
                #job_extra={
                job_extra_directives={
                    '+JobFlavour': '"workday"',
                    },
                #extra = ['--worker-port 10000:10100']
                worker_extra_args = ['--worker-port 10000:10100']
                ) as cluster:
            print(cluster.job_script())
            with Client(cluster) as client:
                futures = []
                cluster.scale(args.njobs)
                for i in range(args.njobs):
                  f = client.submit(hname)
                  futures.append(f)
                print('Result is {}'.format(client.gather(futures)))
                print("Waiting for at least one worker...")
                client.wait_for_workers(1)
                from dask.distributed import performance_report
                client.upload_file('xb.py')
                with performance_report(filename="dask-report.html"):
                    output = processor.run_uproot_job(
                        flist,
                        treename='Events',
                        processor_instance=processor_instance,
                        executor=processor.dask_executor,
                        executor_args={
                            "client": client,
                            #"skipbadfiles": args.skipbadfiles,
                            "schema": processor.NanoAODSchema,
                            "retries": 4,
                        },
                        chunksize=100_000,
                        #maxchunks=args.max,
                    )
                    #save(output, f'/afs/cern.ch/user/b/byates/CMSSW_10_6_18/src/BFrag/BFrag/histos/{outname}.pkl')
                    save(output, f'histos/{outname}.pkl')

    elif executor ==  "work_queue":
        executor = processor.WorkQueueExecutor(**executor_args)
        runner = processor.Runner(executor, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=180)

    #output = runner(flist, treename, processor_instance)
    dt = time.time() - tstart

    if executor == "work_queue":
        print('Processed {} events in {} seconds ({:.2f} evts/sec).'.format(nevts_total,dt,nevts_total/dt))

    #nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    #nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    #print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))

    if executor == "futures":
        print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

    # Save the output
    save(output, f'/afs/cern.ch/user/b/byates/CMSSW_10_6_18/src/BFrag/BFrag/histos/{outname}.pkl')
    #if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    #out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
    #print(f"\nSaving output in {out_pkl_file}...")
    #with gzip.open(out_pkl_file, "wb") as fout:
    #    cloudpickle.dump(output, fout)
    print("Done!")
