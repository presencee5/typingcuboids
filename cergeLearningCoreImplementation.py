import torch as pt
import torch.nn as nn
import pickle as pkl
import numpy as np
import h5py as h5
import multiprocessing as mp
mp.freeze_support()
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pointnet2_ops._ext import furthest_point_sampling as furPntSmpCuda
from ptbkbone.main import get_args_parser, callmain
from detectron2.engine import default_argument_parser




class CubMvrgnPairs(pt.utils.data.Dataset):
    def __init__(self,istrn,rootdir,voxdir=None,useoripirdata=False,npntcmb=8192,fpsofst=5,hpcdvc="cuda:0",bkboneftdir=None,nodrprate=None,
                 partlbl=None,cuboridataonly=True,loadcubfeat=False,loadppcl=False,infridximgdir=None,excludeVwIdForIfr=None):
        self.bolIstrn = istrn

        if istrn:
            self.bolUsOridta = useoripirdata
            self.ptdvcHpc = pt.device(hpcdvc)

            self.dctCubAbs = {}
            self.lstPairs = []

            lstRdTgts = glob(rootdir+"*.npy")
            if not useoripirdata:
                h5filTmp = h5.File(voxdir+"voxs64.hdf5","r")
                aryVoxels = np.array(h5filTmp["voxels"])
                with open(voxdir+"objId.txt","r") as filTmp:
                    lstObjIdxMapping = filTmp.readlines()

            scaI = 0
            for scaI in range(len(lstRdTgts)):
                with open(lstRdTgts[scaI][0:-4]+".pkl","rb") as filTmp:
                    dctTmp = pkl.load(filTmp)

                strObjId = lstRdTgts[scaI].split(rootdir[0:-1])[1][1:-4]

                if useoripirdata:
                    self.dctCubAbs[strObjId] = pt.from_numpy(self.samplePntsOnCubAbs(dctTmp["cubabs"]))
                else:
                    aryObjVoxels = np.stack(np.nonzero(aryVoxels[lstObjIdxMapping.index(strObjId+"\n")]==1)[0:3]).T
                    self.dctCubAbs[strObjId] = self.voxelPartitionByCuboids(aryObjVoxels,self.samplePntsOnCubAbs(dctTmp["cubabs"],nsmpperedg=10))

                scaJ = 0
                if useoripirdata:
                    for scaJ in range(len(dctTmp["cmbbxs"])):
                        self.lstPairs.append({"cmbid":pt.from_numpy(dctTmp["cmbbxs"][scaJ]["cmbid"]).to(pt.uint8),
                                              "objid":strObjId,
                                              "boxes":pt.from_numpy(dctTmp["cmbbxs"][scaJ]["boxes"].astype(np.int32)).to(pt.int16)})
                else:
                    for scaJ in range(len(dctTmp["cmbbxs"])):
                        self.lstPairs.append({"cmbid":pt.from_numpy(dctTmp["cmbbxs"][scaJ]["cmbid"]).to(pt.uint8),
                                              "objid":strObjId})

                print("{} training pairs (with original data: {}) loaded.".format(len(self.lstPairs),useoripirdata))

            if useoripirdata:
                self.scaFpsofst = fpsofst
                self.scaNpntCmb = npntcmb
                self.strMviRootDir = rootdir
            else:
                self.strBknfeatDir = bkboneftdir
                self.lstNodrpratRange = nodrprate

            self.scaLength = len(self.lstPairs)

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Done!")

            del(lstRdTgts,scaI,filTmp,dctTmp,strObjId,scaJ)
            if not useoripirdata:
                del(h5filTmp,lstObjIdxMapping,aryVoxels,aryObjVoxels)

        else:
            self.bolCubOriDataOnly = cuboridataonly
            self.ptdvcHpc = pt.device(hpcdvc)

            if cuboridataonly:
                self.dctCubAbs = {}
                self.lstCubCmbs = []

                lstRdTgts = glob(rootdir+"*.pkl")
                filTmp = None
                scaI = 0
                strObjId = None
                dctTmp = None
                scaJ = 0

                for scaI in range(len(lstRdTgts)):
                    with open(lstRdTgts[scaI],"rb") as filTmp:
                        dctTmp = pkl.load(filTmp)

                    strObjId = lstRdTgts[scaI].split("\\")[-1][0:-4]

                    self.dctCubAbs[strObjId] = pt.from_numpy(self.samplePntsOnCubAbs(dctTmp["cubabs"]))
                    for scaJ in range(len(dctTmp["cmbid"])):
                        self.lstCubCmbs.append({"cmbid":pt.from_numpy(dctTmp["cmbid"][scaJ]).to(pt.uint8),
                                                "objid":strObjId,
                                                "cmbidx":scaJ})

                    print("{} loaded (cuboridataonly: {}).".format(len(self.lstCubCmbs),cuboridataonly))

                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Done!")

                self.scaLength = len(self.lstCubCmbs)
                self.scaFpsofst = fpsofst
                self.scaNpntCmb = npntcmb

                del(lstRdTgts,scaI,filTmp,strObjId,dctTmp,scaJ)

            else:
                self.lstMviRgnsPpcl = []

                lstRdTgts = glob(infridximgdir+"*_Img.npy")
                scaI = 0
                scaJ = 0
                scaK = 0
                filTmp = None
                lstTmp = None
                aryPpcl = None
                lstRemainBxByViewFilt = None

                for scaI in range(len(lstRdTgts)):
                    for scaJ in range(len(partlbl)):
                        with open(lstRdTgts[scaI].split("_Img.npy")[0]+"_"+partlbl[scaJ]+".pkl","rb") as filTmp:
                            lstTmp = pkl.load(filTmp)

                        aryPpcl = np.loadtxt("{}_{}.xyz".format(infridximgdir+lstRdTgts[scaI].split("_Img.npy")[0].split("\\")[-1],partlbl[scaJ]))

                        if len(lstTmp) == 0 or aryPpcl.shape[0] == 0:
                            continue

                        lstRemainBxByViewFilt = []
                        for scaK in range(len(lstTmp)):
                            if not lstTmp[scaK]["image_id"] in excludeVwIdForIfr:
                                lstRemainBxByViewFilt.append(pt.tensor([int(np.floor(lstTmp[scaK]["box"][1])),
                                                                        int(np.floor(lstTmp[scaK]["box"][3])),
                                                                        int(np.floor(lstTmp[scaK]["box"][0])),
                                                                        int(np.floor(lstTmp[scaK]["box"][2])),
                                                                        int(lstTmp[scaK]["image_id"])],
                                                                       dtype=pt.int16))

                        self.lstMviRgnsPpcl.append({"boxes":pt.stack(lstRemainBxByViewFilt),
                                                    "objid":lstRdTgts[scaI].split("_Img.npy")[0].split("\\")[-1],
                                                    "partname":partlbl[scaJ],
                                                    "ppcl":pt.unsqueeze(pt.from_numpy(aryPpcl.astype(np.float32)),0)})

                    print("{} loaded (cuboridataonly: {}).".format(len(self.lstMviRgnsPpcl),cuboridataonly))

                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Done!")

                self.scaLength = len(self.lstMviRgnsPpcl)
                self.strMviRootDir = infridximgdir
                self.strBknfeatDir = bkboneftdir
                self.bolLoadCubFeat = loadcubfeat
                self.bolLoadPpcl = loadppcl

                del(lstRdTgts,filTmp,lstTmp,scaI,scaJ,scaK,aryPpcl,lstRemainBxByViewFilt)


    def __len__(self):
        return self.scaLength


    def __getitem__(self,idx):
        dctPair = None
        tsrCmb = None
        filTmp = None
        lstBknFeat = None
        scaNumBx = None
        dctMviRgnsPpcl = None
        lstBkbnFeatLoadTgt = None
        lstBkbnFeat = None
        scaI = 0
        lstCubAbs = []
        tsrCmbid = None
        lstCmb = []

        if self.bolIstrn:
            dctPair = self.lstPairs[idx]

            if self.bolUsOridta:
                tsrCmb = pt.reshape(self.dctCubAbs[dctPair["objid"]][dctPair["cmbid"].to(int)],(1,-1,3))

                return (idx,
                        dctPair["objid"],
                        tsrCmb[0][furPntSmpCuda(tsrCmb.to(self.ptdvcHpc)+self.scaFpsofst,self.scaNpntCmb)[0].cpu().detach().to(int)],
                        dctPair["boxes"].to(pt.long),
                        pt.from_numpy(np.load(self.strMviRootDir+dctPair["objid"]+".npy")))

            else:
                with open(self.strBknfeatDir+"smp{}.pkl".format(idx),"rb") as filTmp:
                    lstBknFeat = pkl.load(filTmp)
                lstBknFeat[0] = pt.from_numpy(lstBknFeat[0])
                lstBknFeat[1] = pt.from_numpy(lstBknFeat[1])
                scaNumBx = lstBknFeat[1].size(0)
                lstBknFeat[1] = lstBknFeat[1][pt.randperm(scaNumBx)[0:int(np.ceil(scaNumBx*(self.lstNodrpratRange[0]+np.random.rand()*(self.lstNodrpratRange[1]-self.lstNodrpratRange[0]))))],:]
                lstCubAbs = self.dctCubAbs[dctPair["objid"]]
                tsrCmbid = dctPair["cmbid"]
                for scaI in range(tsrCmbid.size(0)):
                    lstCmb.append(lstCubAbs[tsrCmbid[scaI]])

                return (idx,
                        dctPair["objid"],
                        lstBknFeat[0],
                        lstBknFeat[1],
                        np.concatenate(lstCmb,0))

        else:
            if self.bolCubOriDataOnly:
                tsrCmb = self.dctCubAbs[self.lstCubCmbs[idx]["objid"]][self.lstCubCmbs[idx]["cmbid"].to(int)]
                tsrCmb = pt.reshape(tsrCmb,(1,-1,3))

                return (idx,
                        self.lstCubCmbs[idx]["objid"],
                        tsrCmb[0][furPntSmpCuda(tsrCmb.to(self.ptdvcHpc)+self.scaFpsofst,self.scaNpntCmb)[0].cpu().detach().to(int)],
                        self.lstCubCmbs[idx]["cmbidx"])

            else:
                dctMviRgnsPpcl = self.lstMviRgnsPpcl[idx]

                if not self.bolLoadCubFeat and not self.bolLoadPpcl:
                    return (idx,
                            dctMviRgnsPpcl["objid"],
                            None,
                            dctMviRgnsPpcl["boxes"].to(pt.long),
                            pt.from_numpy(np.load(self.strMviRootDir+dctMviRgnsPpcl["objid"]+"_Img.npy")),
                            dctMviRgnsPpcl["partname"],
                            None)

                if not self.bolLoadCubFeat and self.bolLoadPpcl:
                    return (idx,
                            dctMviRgnsPpcl["objid"],
                            None,
                            dctMviRgnsPpcl["boxes"].to(pt.long),
                            pt.from_numpy(np.load(self.strMviRootDir+dctMviRgnsPpcl["objid"]+"_Img.npy")),
                            dctMviRgnsPpcl["partname"],
                            dctMviRgnsPpcl["ppcl"][0][furPntSmpCuda(dctMviRgnsPpcl["ppcl"].to(self.ptdvcHpc),dctMviRgnsPpcl["ppcl"].size(1)//6)[0].cpu().detach().to(int)])

                lstBkbnFeatLoadTgt = glob(self.strBknfeatDir+"test/"+dctMviRgnsPpcl["objid"]+"Cmb*Feat.pkl")

                lstBkbnFeat = []
                scaI = 0
                for scaI in range(len(lstBkbnFeatLoadTgt)):
                    with open(lstBkbnFeatLoadTgt[scaI],"rb") as filTmp:
                        lstBkbnFeat.append(pt.from_numpy(pkl.load(filTmp)[0]))

                return (idx,
                        dctMviRgnsPpcl["objid"],
                        pt.stack(lstBkbnFeat),
                        dctMviRgnsPpcl["boxes"].to(pt.long),
                        pt.from_numpy(np.load(self.strMviRootDir+dctMviRgnsPpcl["objid"]+"_Img.npy")),
                        dctMviRgnsPpcl["partname"],
                        dctMviRgnsPpcl["ppcl"][0][furPntSmpCuda(dctMviRgnsPpcl["ppcl"].to(self.ptdvcHpc),dctMviRgnsPpcl["ppcl"].size(1)//6)[0].cpu().detach().to(int)])


    def samplePntsOnCubAbs(self, cubabs, nsmpperedg=25):
        aryRslt = np.zeros((cubabs.shape[0],nsmpperedg**2*6,3),dtype=np.float32)
        scaI = 0
        for scaI in range(cubabs.shape[0]):
            aryRslt[scaI] = sample_points_on_cube_faces(cubabs[scaI],nsmpperedg)

        return aryRslt


    def voxelPartitionByCuboids(self, voxels, vertices):
        scaNsmpPerCub = vertices.shape[1]

        tsrVoxels = pt.from_numpy(voxels).to(pt.float32).to(self.ptdvcHpc)
        tsrVoxelAlgn = pt.from_numpy(voxels).to(pt.float32).to(self.ptdvcHpc)
        tsrVertices = pt.from_numpy(np.reshape(vertices,(-1,3))).to(self.ptdvcHpc)

        with pt.no_grad():
            tsrBoxMax = pt.max(tsrVoxels,axis=0,keepdim=True)[0]
            tsrBoxMin = pt.min(tsrVoxels,axis=0,keepdim=True)[0]
            tsrVoxels = tsrVoxels - (tsrBoxMax+tsrBoxMin) / 2
            tsrVoxels = tsrVoxels / (pt.sqrt(pt.sum((tsrBoxMax-tsrBoxMin)**2))/2)
            tsrBoxMax = pt.max(tsrVertices,axis=0,keepdim=True)[0]
            tsrBoxMin = pt.min(tsrVertices,axis=0,keepdim=True)[0]
            tsrVertices = tsrVertices / (pt.sqrt(pt.sum((tsrBoxMax-tsrBoxMin)**2))/2)
            tsrVoxelAlgn = tsrVoxelAlgn - pt.round((pt.mean(tsrVoxelAlgn,axis=0,keepdim=True) - pt.tensor([[31.5,31.5,31.5]],device=self.ptdvcHpc)))

            tsrVoxels = pt.unsqueeze(tsrVoxels,1)
            tsrVertices = pt.unsqueeze(tsrVertices,0).expand((tsrVoxels.size(0),tsrVertices.size(0),tsrVertices.size(1)))
            aryAsnToCub = (pt.argmin(pt.sum((tsrVoxels-tsrVertices)**2,dim=2),dim=1) // scaNsmpPerCub).cpu().numpy()

        scaI = 0
        lstRslt = []
        for scaI in range(vertices.shape[0]):
            lstRslt.append(tsrVoxelAlgn.cpu().detach().numpy()[np.nonzero(aryAsnToCub==scaI)[0]].astype(np.float32))

        del(tsrBoxMax,tsrBoxMin,aryAsnToCub)
        pt.cuda.empty_cache()
        return lstRslt


    def setLoadCubFeat(self, option):
        if hasattr(self,"bolLoadCubFeat"):
            self.bolLoadCubFeat = option
        else:
            print("No attribute `bolCubOriDataOnly`. Nothing done.")

    def setLoadPpcl(self, option):
        if hasattr(self,"bolLoadPpcl"):
            self.bolLoadPpcl = option
        else:
            print("No attribute `bolLoadPpcl`. Nothing done.")



def collateFnTb(samples):
    tplSmp = None
    lstRslt = [[],[],[],[],[]]

    for tplSmp in samples:
        lstRslt[0].append(tplSmp[0])
        lstRslt[1].append(tplSmp[1])
        lstRslt[2].append(tplSmp[2])
        lstRslt[3].append(tplSmp[3])
        lstRslt[4].append(tplSmp[4])

    return lstRslt



def collateFnSb(samples):
    tplSmp = None
    lstRslt = [[],[],[],[]]

    for tplSmp in samples:
        lstRslt[0].append(tplSmp[0])
        lstRslt[1].append(tplSmp[1])
        lstRslt[2].append(tplSmp[2])
        lstRslt[3].append(tplSmp[3])

    return lstRslt



def collateFnTt(samples):
    tplSmp = None
    lstRslt = [[],[],[],[],[],[]]
    tsrSqLen = None
    tsrPadMsk = None

    for tplSmp in samples:
        lstRslt[0].append(tplSmp[0])
        lstRslt[1].append(tplSmp[4])
        if tplSmp[3].size(0) == 0:
            lstRslt[2].append(-1e-7+pt.rand((512,))*2e-7)
            lstRslt[3].append(-1e-7+pt.rand((1,1024))*2e-7)
        else:
            lstRslt[2].append(tplSmp[2])
            lstRslt[3].append(tplSmp[3])
        lstRslt[5].append(tplSmp[4])

    with pt.no_grad():
        lstRslt[2] = pt.stack(lstRslt[2],dim=0)
        tsrSqLen = pt.unsqueeze(pt.tensor([tsrI.size(0) for tsrI in lstRslt[3]],dtype=pt.int16),dim=1)
        lstRslt[3] = nn.utils.rnn.pad_sequence(lstRslt[3],batch_first=True)
        tsrPadMsk = pt.arange(lstRslt[3].size(1)).expand(lstRslt[3].size(0), lstRslt[3].size(1)) >= tsrSqLen

    scaBatchSz = len(lstRslt[1])

    if nmspArgsMjr.scaLosIntrWei == 0:
        lstRslt[1] = pt.zeros((scaBatchSz,scaBatchSz),dtype=bool)
        lstRslt[4] = tsrPadMsk
        return lstRslt

    arySimilarShp = np.zeros((scaBatchSz,scaBatchSz),dtype=bool)
    arySimilarCntrid = np.zeros((scaBatchSz,scaBatchSz),dtype=np.float32)

    lstTasks = [(scaI,scaJ,lstRslt[1][scaI],lstRslt[1][scaJ],nmspArgsMjr.scaLosIntrShpThres) for scaI in range(scaBatchSz) for scaJ in range(scaI+1,scaBatchSz)]
    mppool = None
    lstMpRslts = None
    tplMpRslts = None
    with mp.Pool(processes=nmspArgsMjr.scaLosIntrCalcProc) as mppool:
        lstMpRslts = mppool.map(fillSimilarShpAndCntridMsk,lstTasks)
    for tplMpRslts in lstMpRslts:
        arySimilarShp[tplMpRslts[0],tplMpRslts[1]] = tplMpRslts[2]
        arySimilarShp[tplMpRslts[1],tplMpRslts[0]] = tplMpRslts[2]
        arySimilarCntrid[tplMpRslts[0],tplMpRslts[1]] = tplMpRslts[3]
        arySimilarCntrid[tplMpRslts[1],tplMpRslts[0]] = tplMpRslts[3]

    aryValdSimilarCentrid = arySimilarCntrid != -1
    scaMinDist = np.min(arySimilarCntrid[aryValdSimilarCentrid])
    scaMaxDist = np.max(arySimilarCntrid[aryValdSimilarCentrid])
    arySimilarCntrid = ((arySimilarCntrid - scaMinDist) / (scaMaxDist - scaMinDist)) < nmspArgsMjr.scaLosIntrPosThres
    arySimilarCntrid = np.logical_and(arySimilarCntrid,aryValdSimilarCentrid)

    lstRslt[1] = pt.from_numpy(np.logical_and(arySimilarShp,arySimilarCntrid))

    lstRslt[4] = tsrPadMsk

    return lstRslt



def collateFnSmode(samples):
    tplSmp = None
    lstRslt = [[],[],[],[],[],[],[]]

    for tplSmp in samples:
        lstRslt[0].append(tplSmp[0])
        lstRslt[1].append(tplSmp[1])
        lstRslt[2].append(tplSmp[2])
        lstRslt[3].append(tplSmp[3])
        lstRslt[4].append(tplSmp[4])
        lstRslt[5].append(tplSmp[5])
        lstRslt[6].append(tplSmp[6])

    return lstRslt



def fillSimilarShpAndCntridMsk(args):
    if args[2].shape[0] == 0 or args[3].shape[0] == 0:
        return args[0], args[1], False, -1

    aryCentroid1 = np.mean(args[2],axis=0,keepdims=True)
    aryCentroid2 = np.mean(args[3],axis=0,keepdims=True)

    aryVoxels1 = np.floor(args[2] - (aryCentroid1 - np.array([[31.5,31.5,31.5]]))).astype(np.int32)
    aryVoxels2 = np.floor(args[3] - (aryCentroid2 - np.array([[31.5,31.5,31.5]]))).astype(np.int32)
    aryVoxels1 = aryVoxels1[:,0]*64**2 + aryVoxels1[:,1]*64 + aryVoxels1[:,2]
    aryVoxels2 = aryVoxels2[:,0]*64**2 + aryVoxels2[:,1]*64 + aryVoxels2[:,2]
    aryIntersect = np.intersect1d(aryVoxels1,aryVoxels2,True)
    aryUnion = np.union1d(aryVoxels1,aryVoxels2)

    if aryIntersect.shape[0]/aryUnion.shape[0] > args[4]:
        return args[0], args[1], True, np.sum((aryCentroid1 - aryCentroid2)**2)

    return args[0], args[1], False, np.sum((aryCentroid1 - aryCentroid2)**2)



def sample_points_on_cube_faces(cube_vertices, n_samples_per_edge=20):
    v0, v1, v2, v3, v4, v5, v6, v7 = cube_vertices

    faces = [
        [v0, v1, v3, v2],
        [v4, v6, v7, v5],
        [v0, v1, v5, v4],
        [v2, v3, v7, v6],
        [v0, v2, v6, v4],
        [v1, v3, v7, v5],
    ]

    sampled_points = []

    for face in faces:
        A, B, C, D = face

        u = np.linspace(0, 1, n_samples_per_edge).astype(np.float32)
        v = np.linspace(0, 1, n_samples_per_edge).astype(np.float32)
        uu, vv = np.meshgrid(u, v)

        points = (
            (1 - uu)[:, :, np.newaxis] *
            (1 - vv)[:, :, np.newaxis] * A   # (1-u)(1-v)A
            + uu[:, :, np.newaxis] *
            (1 - vv)[:, :, np.newaxis] * B       # u(1-v)B
            + uu[:, :, np.newaxis] * vv[:, :, np.newaxis] * C             # uvC
            + (1 - uu)[:, :, np.newaxis] *
            vv[:, :, np.newaxis] * D       # (1-u)vD
        )

        sampled_points.append(points.reshape(-1, 3))

    aryRslt = np.vstack(sampled_points)
    aryRslt += np.random.rand(aryRslt.shape[0],aryRslt.shape[1]) * 5e-3

    return aryRslt



class AlignedFeaEx(nn.Module):
    def __init__(self,mode,argdict):
        super(AlignedFeaEx, self).__init__()

        if mode == "b" or mode == "p":
            self.nmspArgsRgnclp = argdict["argsrgnclp"]
            self.nmspArgsUlip = argdict["argsulip"]

        elif mode == "t":
            self._createHead({"indimrgn":argdict["indimrgn"],
                              "indimcmb":argdict["indimcmb"],
                              "mhaheadnum":argdict["mhaheadnum"],
                              "algndim":argdict["algndim"]})

        elif mode == "s":
            self.nmspArgsRgnclp = argdict["argsrgnclp"]
            self.nmspArgsUlip = argdict["argsulip"]
            self._createHead({"indimrgn":argdict["indimrgn"],
                              "indimcmb":argdict["indimcmb"],
                              "mhaheadnum":argdict["mhaheadnum"],
                              "algndim":argdict["algndim"]})

        self.strMode = mode


    def _createHead(self,argdict):
        self.ptnnRgnMha = nn.MultiheadAttention(argdict["indimrgn"],argdict["mhaheadnum"],batch_first=True)
        # self.ptnnRgnMha1 = nn.MultiheadAttention(argdict["indimrgn"],argdict["mhaheadnum"],batch_first=True)
        # self.ptnnRgnMha2 = nn.MultiheadAttention(argdict["indimrgn"],argdict["mhaheadnum"],batch_first=True)
        # self.ptnnRgnMha3 = nn.MultiheadAttention(argdict["indimrgn"],argdict["mhaheadnum"],batch_first=True)
        self.ptnnRgnLin = nn.Linear(argdict["indimrgn"],argdict["algndim"])
        self.ptnsqCubcmbHead = nn.Sequential(nn.Linear(argdict["indimcmb"],argdict["indimcmb"]),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(argdict["indimcmb"],argdict["indimcmb"]),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(argdict["indimcmb"],argdict["indimcmb"]),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(argdict["indimcmb"],argdict["algndim"]))


    def runBkbone(self,dtaldr,dtastlen,svedir=None):
        return callmain(self.nmspArgsUlip,self.nmspArgsRgnclp,dtaldr,dtastlen,svedir)


    def forward(self,xCmb,xRgn,xRgnSqPadMsk=None):
        if self.strMode == "b" or self.strMode == "p":
            print("`forward` is not available in the `b` or `p` mode.")
            exit(1)

        elif self.strMode == "t" or self.strMode == "s":
            tsrFeatCmb = self.ptnsqCubcmbHead(xCmb)

            tplFeatRgnRes = self.ptnnRgnMha(xRgn,xRgn,xRgn,xRgnSqPadMsk) # 1(xRgn,xRgn,xRgn,xRgnSqPadMsk)
            tsrFeatRgn = tplFeatRgnRes[0] + xRgn
            # tplFeatRgnRes = self.ptnnRgnMha2(tsrFeatRgn,tsrFeatRgn,tsrFeatRgn,xRgnSqPadMsk)
            # tsrFeatRgn = tplFeatRgnRes[0] + tsrFeatRgn
            # tplFeatRgnRes = self.ptnnRgnMha3(tsrFeatRgn,tsrFeatRgn,tsrFeatRgn,xRgnSqPadMsk)
            # tsrFeatRgn = tplFeatRgnRes[0] + tsrFeatRgn
            tsrFeatRgn = nn.functional.layer_norm(tsrFeatRgn,[tsrFeatRgn.size(2)])
            if xRgnSqPadMsk is None:
                tsrFeatRgn = pt.mean(tsrFeatRgn,dim=1)
            else:
                tsrFeatRgn = pt.sum(tsrFeatRgn,dim=1) / pt.sum((~xRgnSqPadMsk).to(int),dim=1,keepdim=True) # (bs,c) / (bs,1)
            tsrFeatRgn = self.ptnnRgnLin(tsrFeatRgn)

            return (tsrFeatCmb,tsrFeatRgn,tplFeatRgnRes[1])



def getMjrArgPrsr():
    argpsr = ArgumentParser()
    argpsr.add_argument("--strMode",choices=["b","t","p","s"],type=str)
    argpsr.add_argument("--strDtarotDir",type=str,default="../glippass/output/")
    argpsr.add_argument("--strOrivoxDir",type=str,default="../dataset/output/")
    argpsr.add_argument("--strFeatDir",type=str,default="./presavedFeaturesByBkbone/")
    argpsr.add_argument("--strCkptDir",type=str,default="./trndCkpts/")
    argpsr.add_argument("--strCategryId",type=str)
    argpsr.add_argument("--strCkptSvName",type=str,default="")
    argpsr.add_argument("--strCkptLdName",type=str,default="")
    argpsr.add_argument("--bolCkptStart",action="store_true")
    argpsr.add_argument("--lstNodrpratRange",nargs=2,type=float,default=[0.95,0.99])
    argpsr.add_argument("--scaRgnInDim",type=int,default=1024)
    argpsr.add_argument("--scaCmbInDim",type=int,default=512)
    argpsr.add_argument("--scaAlgnDim",type=int,default=512)
    argpsr.add_argument("--scaMhaHead",type=int,default=8)
    argpsr.add_argument("--scaLosIntrWei",type=float,default=0.0)
    argpsr.add_argument("--scaLosIntrCalcProc",type=int,default=6)
    argpsr.add_argument("--scaLosIntrPosThres",type=float,default=0.1)
    argpsr.add_argument("--scaLosIntrShpThres",type=float,default=0.5)
    argpsr.add_argument("--scaCalbtchsz",type=int,default=1)
    argpsr.add_argument("--scaAcutim",type=int,default=1)
    argpsr.add_argument("--lstLr",nargs=2,type=float,default=[0.0001,0.0001])
    argpsr.add_argument("--scaLrDecyStp",type=int,default=100)
    argpsr.add_argument("--scaLrDecyGma",type=float,default=0.5)
    argpsr.add_argument("--scaEpoch",type=int,default=36)
    argpsr.add_argument("--lstEpcToSave",nargs="*",type=int,default=[0,5,10,15,20,25,30,35])
    argpsr.add_argument("--strDevice",type=str,default="cuda:0")
    argpsr.add_argument("--lstPartLabel",nargs="*",type=str,default=[])
    argpsr.add_argument("--strInfrOutDir",type=str,default="")
    argpsr.add_argument("--strInfrCabsDir",type=str,default="../testLocalCbabsEnum/output/")
    argpsr.add_argument("--strInfrIdxImgDir",type=str,default="../glippass/output/")
    argpsr.add_argument("--scaInfrPoolSz",type=int,default=0)
    argpsr.add_argument("--lstInfrExcludeVw",nargs="*",type=int,default=[])

    return argpsr



def main():
    nmspArgsRgnclp.opts = ['MODEL.WEIGHTS', './ptbkbone/regionclip/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth',
                           'MODEL.CLIP.CROP_REGION_TYPE', 'RPN',
                           'MODEL.CLIP.MULTIPLY_RPN_SCORE', 'True',
                           'MODEL.CLIP.OFFLINE_RPN_CONFIG', './ptbkbone/regionclip/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
                           'MODEL.CLIP.BB_RPN_WEIGHTS', './ptbkbone/regionclip/pretrained_ckpt/rpn/rpn_lvis_866.pth',
                           'INPUT_DIR', './ptbkbone/regionclip/datasets/custom_images',
                           'OUTPUT_DIR', './ptbkbone/regionclip/output/region_feats',
                           'MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST', '2']

    if nmspArgsMjr.strMode != "p" and nmspArgsMjr.strMode != "s":
        train(nmspArgsRgnclp,nmspArgsUlip,nmspArgsMjr)
    elif nmspArgsMjr.strMode == "p":
        preExtractCubFeatOnly(nmspArgsRgnclp,nmspArgsUlip,nmspArgsMjr)
    elif nmspArgsMjr.strMode == "s":
        inference(nmspArgsRgnclp,nmspArgsUlip,nmspArgsMjr)



def loss(input, tmptr, sameId, sameIdWei):
    scaBs = input.size(0)//2

    tsrSameId = sameId
    tsrSameId[pt.eye(scaBs,dtype=bool).to(input.device)] = False
    tsrSameId = tsrSameId.detach()

    tsrSameIdLrg = pt.cat((pt.cat((tsrSameId.to(int), pt.zeros((scaBs,scaBs),dtype=int,device=input.device)),dim=1),
                           pt.cat((pt.zeros((scaBs,scaBs),dtype=int,device=input.device), tsrSameId.to(int)),dim=1)),
                          dim=0)
    tsrSameIdLrg = pt.reshape(tsrSameIdLrg[~pt.eye(2*scaBs,dtype=bool).to(input.device)],[scaBs*2,scaBs*2-1])
    tsrSameIdLrg = tsrSameIdLrg.detach()

    tsrApplied = pt.eye(scaBs,dtype=bool).to(input.device).repeat(2,2)
    tsrApplied[0:scaBs,0:scaBs] = tsrSameId
    tsrApplied[scaBs:2*scaBs,scaBs:2*scaBs] = tsrSameId
    tsrApplied = pt.reshape(tsrApplied[~pt.eye(2*scaBs,dtype=bool).to(input.device)],[scaBs*2,scaBs*2-1])
    tsrApplied = tsrApplied.detach()

    tsrIntraModalPairs = pt.eye(scaBs,dtype=bool).to(input.device).repeat(2,2)
    tsrIntraModalPairs = pt.reshape(tsrIntraModalPairs[~pt.eye(2*scaBs,dtype=bool).to(input.device)],[scaBs*2,scaBs*2-1])
    tsrIntraModalPairs = tsrIntraModalPairs.detach()

    tsrSim = pt.matmul(input,pt.t(input).contiguous())
    tsrLos = tsrSim / tmptr
    tsrLos = nn.functional.log_softmax(pt.reshape(tsrLos[~pt.eye(2*scaBs,dtype=bool).to(input.device)],[scaBs*2,scaBs*2-1]),dim=1)
    tsrLos = tsrLos * (tsrSameIdLrg*(sameIdWei-1) + 1)

    return pt.mean(-tsrLos[tsrApplied]), tsrSim, pt.sum(tsrSameId.to(int)) / (tsrSameId.size(0)**2), pt.mean(-tsrLos[tsrIntraModalPairs]).item()



def compareBboxAndCentrd(denvx1,denvx2,dvc):
    if denvx1.shape[0] == 0 or denvx2.shape[0] == 0:
        return None, None

    tsrDenvx1 = pt.from_numpy(denvx1).to(dvc).to(int)
    tsrDenvx2 = pt.from_numpy(denvx2).to(dvc).to(int)

    with pt.no_grad():
        tsrMin1 = pt.min(tsrDenvx1,dim=0)[0].to(pt.float32)
        tsrMax1 = pt.max(tsrDenvx1,dim=0)[0].to(pt.float32)
        tsrMin2 = pt.min(tsrDenvx2,dim=0)[0].to(pt.float32)
        tsrMax2 = pt.max(tsrDenvx2,dim=0)[0].to(pt.float32)

        tsrDenvx1 = tsrDenvx1.to(pt.float32)
        tsrDenvx2 = tsrDenvx2.to(pt.float32)
        tsrCentrd1 = pt.mean(tsrDenvx1,dim=0)
        tsrCentrd2 = pt.mean(tsrDenvx2,dim=0)

        return pt.mean(pt.cat((pt.abs(tsrMin1-tsrMin2),pt.abs(tsrMax1-tsrMax2)))).item(), pt.sqrt(pt.sum((tsrCentrd1-tsrCentrd2)**2)).item()



def train(argsrgnclp,argsulip,argsmjr):
    if argsmjr.strMode == "b":
        print("`b` mode.\n===============================")
        print("Backbones are forced to run on CUDA. CUDA is also used by default for some other operations (like furthest point sampling).")

        ptdast = CubMvrgnPairs(True,
                               argsmjr.strDtarotDir+argsmjr.strCategryId+"/",
                               useoripirdata=True)
        ptdldr = pt.utils.data.DataLoader(ptdast,
                                          argsmjr.scaCalbtchsz*argsmjr.scaAcutim,
                                          shuffle=False,
                                          collate_fn=collateFnTb,
                                          num_workers=0)
        ptnn = AlignedFeaEx(argsmjr.strMode, {"argsrgnclp":argsrgnclp,"argsulip":argsulip})

        ptnn.runBkbone(ptdldr,len(ptdast),argsmjr.strFeatDir+argsmjr.strCategryId+"/")

        print("Done.")

        del(ptdast,ptdldr,ptnn)


    elif argsmjr.strMode == "t":
        print("`t` mode.\n===============================")

        ptdvc = pt.device(argsmjr.strDevice)

        ptdast = CubMvrgnPairs(True,
                               argsmjr.strDtarotDir+argsmjr.strCategryId+"/",
                               voxdir=argsmjr.strOrivoxDir+argsmjr.strCategryId+"Final/train/",
                               bkboneftdir=argsmjr.strFeatDir+argsmjr.strCategryId+"/",
                               nodrprate=argsmjr.lstNodrpratRange)
        ptdldr = pt.utils.data.DataLoader(ptdast,
                                          argsmjr.scaCalbtchsz*argsmjr.scaAcutim,
                                          shuffle=True,
                                          collate_fn=collateFnTt,
                                          num_workers=0,
                                          drop_last=True)

        ptnn = AlignedFeaEx(argsmjr.strMode, {"indimrgn":argsmjr.scaRgnInDim,
                                              "indimcmb":argsmjr.scaCmbInDim,
                                              "algndim":argsmjr.scaAlgnDim,
                                              "mhaheadnum":argsmjr.scaMhaHead}).to(ptdvc)

        ptoptim = Adam([{"params":ptnn.ptnsqCubcmbHead.parameters(),"lr":argsmjr.lstLr[0]},
                        {"params":ptnn.ptnnRgnMha.parameters(),"lr":argsmjr.lstLr[1]},
                        # {"params":ptnn.ptnnRgnMha1.parameters(),"lr":argsmjr.lstLr[1]},
                        # {"params":ptnn.ptnnRgnMha2.parameters(),"lr":argsmjr.lstLr[1]},
                        # {"params":ptnn.ptnnRgnMha3.parameters(),"lr":argsmjr.lstLr[1]},
                        {"params":ptnn.ptnnRgnLin.parameters(),"lr":argsmjr.lstLr[1]}])
        ptlrschr = StepLR(ptoptim,argsmjr.scaLrDecyStp,argsmjr.scaLrDecyGma)

        tplTmp = None
        dctCkpt = None
        tplDtaBch = None
        tplOut = None
        tsrOutCch = None
        tsrOutCchGrad = None
        tplLos = None
        scaLosAcu = 0
        scaIntraModDist = 0
        tsrTopOneMatch = None
        scaTopOneAccAcu = 0
        scaTopOneErrBoxAcu = 0
        scaTopOneErrCntAcu = 0
        scaTopOneErrValidCount = 0
        tplTopOneErr = None
        scaInterPortInBatchAcu = 0
        tsrMatchList = None
        scaJ = None
        scaI = 0

        if argsmjr.bolCkptStart:
            dctCkpt = pt.load(argsmjr.strCkptDir+argsmjr.strCategryId+"/"+argsmjr.strCkptLdName+"_current.pth",map_location=ptdvc)
            ptnn.load_state_dict(dctCkpt["model"])
            ptoptim.load_state_dict(dctCkpt["optim"])
            ptlrschr.load_state_dict(dctCkpt["lrschdulr"])
            scaI = dctCkpt["startEpc"]
            print(">>>>>Checkpoint loaded.\n")

        while scaI < argsmjr.scaEpoch:
            print("Epoch {} with learning rate {}.".format(scaI,[group['lr'] for group in ptoptim.param_groups]))
            ptnn.train()
            ptoptim.zero_grad()
            scaLosAcu = 0
            scaIntraModDist = 0
            scaTopOneAccAcu = 0
            scaInterPortInBatchAcu = 0
            scaTopOneErrBoxAcu = 0
            scaTopOneErrCntAcu = 0
            scaTopOneErrValidCount = 0

            for tplTmp in tqdm(enumerate(ptdldr),ncols=50):
                tplDtaBch = tplTmp[1]

                tsrOutCch = pt.zeros((2, argsmjr.scaAcutim*argsmjr.scaCalbtchsz, argsmjr.scaAlgnDim),
                                     device=ptdvc,
                                     dtype=pt.float32)
                with pt.no_grad():
                    for scaJ in range(argsmjr.scaAcutim):
                        tplOut = ptnn(tplDtaBch[2][argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)].to(ptdvc),
                                      tplDtaBch[3][argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)].to(ptdvc),
                                      tplDtaBch[4][argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)].to(ptdvc))
                        tsrOutCch[0, argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)] = tplOut[0]
                        tsrOutCch[1, argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)] = tplOut[1]

                tsrOutCch.requires_grad_()
                tplLos = loss(pt.flatten(tsrOutCch,0,1),0.5,tplDtaBch[1].to(ptdvc),argsmjr.scaLosIntrWei)
                tplLos[0].backward() # tsrOutCch.grad: dl/df
                with pt.no_grad():
                    tsrOutCchGrad = pt.reshape(tsrOutCch.grad,
                                               (2, argsmjr.scaAcutim*argsmjr.scaCalbtchsz, argsmjr.scaAlgnDim))

                for scaJ in range(argsmjr.scaAcutim):
                    tplOut = ptnn(tplDtaBch[2][argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)].to(ptdvc),
                                  tplDtaBch[3][argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)].to(ptdvc),
                                  tplDtaBch[4][argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)].to(ptdvc))
                    (pt.sum(tsrOutCchGrad[0, argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)] * tplOut[0])+\
                      pt.sum(tsrOutCchGrad[1, argsmjr.scaCalbtchsz * scaJ : argsmjr.scaCalbtchsz * (scaJ+1)] * tplOut[1])).backward()

                ptoptim.step()
                ptoptim.zero_grad()

                scaLosAcu += tplLos[0].item()
                scaIntraModDist += tplLos[3]

                with pt.no_grad():
                    tsrMatchList = pt.max(tplLos[1][len(tplDtaBch[0]):2*len(tplDtaBch[0]),0:len(tplDtaBch[0])],dim=1)[1]
                    tsrTopOneMatch = (tsrMatchList == pt.arange(len(tplDtaBch[0]),device=ptdvc))
                    scaTopOneAccAcu += pt.sum(tsrTopOneMatch.to(int))/len(tplDtaBch[0])
                    for scaJ in range(tsrTopOneMatch.size(0)):
                        if not tsrTopOneMatch[scaJ]:
                            tplTopOneErr = compareBboxAndCentrd(tplDtaBch[5][scaJ],tplDtaBch[5][tsrMatchList[scaJ]],ptdvc)
                            if tplTopOneErr[0] is None or tplTopOneErr[1] is None:
                                continue
                            else:
                                scaTopOneErrBoxAcu += tplTopOneErr[0]
                                scaTopOneErrCntAcu += tplTopOneErr[1]
                                scaTopOneErrValidCount += 1
                scaInterPortInBatchAcu += tplLos[2]

            print("Epoch loss: {}.\n".format(scaLosAcu/(tplTmp[0]+1))+\
                   "Batch-wise top training acc: {}. Batch-wise top box/cnt. err: {}/{}.\n".format(scaTopOneAccAcu/(tplTmp[0]+1),
                                                                                                   scaTopOneErrBoxAcu/scaTopOneErrValidCount,
                                                                                                   scaTopOneErrCntAcu/scaTopOneErrValidCount)+\
                    "Inter-pulling rate: {}. Intra mod. dist: {}\n".format(scaInterPortInBatchAcu/(tplTmp[0]+1),scaIntraModDist/(tplTmp[0]+1)))

            ptlrschr.step()
            scaI += 1

            pt.save({"startEpc":scaI,
                     "model":ptnn.state_dict(),
                     "optim":ptoptim.state_dict(),
                     "lrschdulr":ptlrschr.state_dict()},
                    argsmjr.strCkptDir+argsmjr.strCategryId+"/"+argsmjr.strCkptSvName+"_current.pth")

            if scaI-1 in argsmjr.lstEpcToSave:
                pt.save({"model":ptnn.state_dict()},
                        argsmjr.strCkptDir+argsmjr.strCategryId+"/"+argsmjr.strCkptSvName+"_epc"+str(scaI-1)+".pth")

        del(ptdvc, ptdast, ptdldr, ptnn, ptoptim, ptlrschr, tplTmp, dctCkpt, tplDtaBch, tplOut, tsrOutCch,
            tsrOutCchGrad, tplLos, scaLosAcu, tsrTopOneMatch, scaTopOneAccAcu, scaJ, scaI, scaIntraModDist,
            scaInterPortInBatchAcu, tsrMatchList, scaTopOneErrBoxAcu, scaTopOneErrCntAcu, tplTopOneErr, scaTopOneErrValidCount)



def preExtractCubFeatOnly(argsrgnclp,argsulip,argsmjr):
    print("`p` mode.\n===============================")
    print("The backbone is forced to run on CUDA. CUDA is also used by default for some other operations (like furthest point sampling).")

    ptdast = CubMvrgnPairs(False,
                           argsmjr.strDtarotDir+argsmjr.strCategryId+"/")
    ptdldr = pt.utils.data.DataLoader(ptdast,
                                      argsmjr.scaCalbtchsz*argsmjr.scaAcutim,
                                      shuffle=False,
                                      collate_fn=collateFnSb,
                                      num_workers=0)

    ptnn = AlignedFeaEx(argsmjr.strMode, {"argsrgnclp":argsrgnclp,"argsulip":argsulip})

    ptnn.runBkbone(ptdldr,len(ptdast),argsmjr.strFeatDir+argsmjr.strCategryId+"/test/")

    print("Done.")



def inference(argsrgnclp,argsulip,argsmjr):
    print("`s` mode.\n===============================")
    print("The backbone is forced to run on CUDA. CUDA is also used by default for some other operations (like furthest point sampling).")


    ptdvc = pt.device(argsmjr.strDevice)

    ptdast = CubMvrgnPairs(False,
                           argsmjr.strDtarotDir+argsmjr.strCategryId+"/",
                           bkboneftdir=argsmjr.strFeatDir+argsmjr.strCategryId+"/",
                           partlbl=argsmjr.lstPartLabel,
                           cuboridataonly=False,
                           infridximgdir=argsmjr.strInfrIdxImgDir+argsmjr.strCategryId+"/",
                           excludeVwIdForIfr=argsmjr.lstInfrExcludeVw)
    ptdldrForMvirgnFeatEx = pt.utils.data.DataLoader(ptdast,
                                                     argsmjr.scaCalbtchsz*argsmjr.scaAcutim,
                                                     shuffle=False,
                                                     collate_fn=collateFnSmode,
                                                     num_workers=0)
    ptdldr = pt.utils.data.DataLoader(ptdast,
                                      1,
                                      shuffle=False,
                                      collate_fn=collateFnSmode,
                                      num_workers=0)

    ptnn = AlignedFeaEx(argsmjr.strMode, {"argsrgnclp":argsrgnclp,
                                          "argsulip":argsulip,
                                          "indimrgn":argsmjr.scaRgnInDim,
                                          "indimcmb":argsmjr.scaCmbInDim,
                                          "algndim":argsmjr.scaAlgnDim,
                                          "mhaheadnum":argsmjr.scaMhaHead})

    if argsmjr.scaInfrPoolSz >= 1:
        tsrMviRgnFeats = ptnn.runBkbone(ptdldrForMvirgnFeatEx,len(ptdast))
        ptdast.setLoadCubFeat(True)
        ptdast.setLoadPpcl(True)

    if argsmjr.scaInfrPoolSz < -1:
        ptdast.setLoadPpcl(True)

    ptnn = ptnn.to(ptdvc)

    dctCkpt = pt.load(argsmjr.strCkptDir+argsmjr.strCategryId+"/"+argsmjr.strCkptLdName,map_location=ptdvc)
    ptnn.load_state_dict(dctCkpt["model"])
    print(">>>>>Checkpoint loaded.\n")
    ptnn.eval()

    tplTmp = None
    tplDta = None
    tsrMviRgnFeat = None
    tplOut = None
    tsrInerProduct = None
    lstTopk = []
    filTmp = None
    dctCabsAndEnum = None
    aryCabsPnts = None
    dctCubSelecId = {}
    scaI = 0
    aryCdsForMatch = None
    scaSelectRsltId = None
    strCurrentObjId = ""
    for tplTmp in tqdm(enumerate(ptdldr),ncols=50,position=0):
        tplDta = tplTmp[1]

        with pt.no_grad():
            with open(argsmjr.strInfrCabsDir+argsmjr.strCategryId+"/"+tplDta[1][0]+".pkl","rb") as filTmp:
                dctCabsAndEnum = pkl.load(filTmp)
            aryCabsPnts = ptdast.samplePntsOnCubAbs(dctCabsAndEnum["cubabs"],10)

            if argsmjr.scaInfrPoolSz >= 1:
                tsrMviRgnFeat = tsrMviRgnFeats[pt.nonzero((tsrMviRgnFeats[:,1024]==tplDta[0][0]),as_tuple=True)[0]]

                tplOut = ptnn(tplDta[2][0].to(ptdvc),pt.unsqueeze(tsrMviRgnFeat[:,0:1024],dim=0).to(ptdvc))

                tsrInerProduct = pt.sum(tplOut[1] * tplOut[0],1)

                if tsrInerProduct.size(0) >= argsmjr.scaInfrPoolSz:
                    lstTopk = pt.topk(tsrInerProduct,argsmjr.scaInfrPoolSz)[1].cpu().detach().numpy().tolist()
                else:
                    lstTopk = pt.topk(tsrInerProduct,tsrInerProduct.size(0))[1].cpu().detach().numpy().tolist()

            lstTopk = pt.randperm(2**dctCabsAndEnum["cubabs"].shape[0]-2)[0:np.abs(argsmjr.scaInfrPoolSz)].cpu().detach().numpy().tolist()

            aryCdsForMatch = np.zeros((np.abs(argsmjr.scaInfrPoolSz),)) + 1e8
            dctCubSelecId = {}
            if argsmjr.scaInfrPoolSz != -1:
                for scaI in tqdm(range(len(lstTopk)),position=1,leave=False,ncols=50):
                    aryCdsForMatch[scaI] = chamferDistPytorch(pt.from_numpy(np.reshape(aryCabsPnts[dctCabsAndEnum["cmbid"][lstTopk[scaI]].astype(int)],(-1,3))).to(ptdvc), 
                                                              tplDta[6][0].to(ptdvc))
                    np.savetxt(argsmjr.strInfrOutDir+argsmjr.strCategryId+"/"+tplDta[1][0]+"_"+tplDta[5][0]+"Top"+str(scaI)+".xyz",
                               np.reshape(aryCabsPnts[dctCabsAndEnum["cmbid"][lstTopk[scaI]].astype(int)],(-1,3)),fmt='%.6f')
            scaSelectRsltId = np.argmin(aryCdsForMatch)
            dctCubSelecId[0] = dctCabsAndEnum["cmbid"][lstTopk[scaSelectRsltId]].astype(int)
            np.savetxt(argsmjr.strInfrOutDir+argsmjr.strCategryId+"/"+tplDta[1][0]+"_"+tplDta[5][0]+"Selected.xyz",
                       np.reshape(aryCabsPnts[dctCabsAndEnum["cmbid"][lstTopk[scaSelectRsltId]].astype(int)],(-1,3)),fmt='%.6f')

            if argsmjr.scaInfrPoolSz != -1:
                np.savetxt(argsmjr.strInfrOutDir+argsmjr.strCategryId+"/"+tplDta[1][0]+"_"+tplDta[5][0]+"Ppcl.xyz",
                           tplDta[6][0].cpu().detach().numpy(),fmt='%.6f')

            with open(argsmjr.strInfrOutDir+argsmjr.strCategryId+"/"+tplDta[1][0]+"_"+tplDta[5][0]+".pkl", "wb") as filTmp:
                pkl.dump(dctCubSelecId,filTmp)

            if argsmjr.scaInfrPoolSz >= 1:
                np.save(argsmjr.strInfrOutDir+argsmjr.strCategryId+"/"+tplDta[1][0]+"_"+tplDta[5][0]+".npy",
                        tsrInerProduct.cpu().detach().numpy())

                if tplDta[1][0] != strCurrentObjId:
                    strCurrentObjId = tplDta[1][0]
                    np.save(argsmjr.strInfrOutDir+argsmjr.strCategryId+"/"+tplDta[1][0]+"allCubFeatures.npy",
                            tplOut[0].cpu().detach().numpy())



def chamferDistPytorch(pc1, pc2):
    with pt.no_grad():
        pc1_expanded = pc1.unsqueeze(1)  # (n, 1, 3)
        pc2_expanded = pc2.unsqueeze(0)  # (1, m, 3)

        dists = pt.sum((pc1_expanded - pc2_expanded) ** 2, dim=2)  # (n, m)
        min_dist1 = pt.min(dists, dim=1)[0]  # (n,)
        min_dist2 = pt.min(dists, dim=0)[0]  # (m,)
        chamfer_dist = pt.mean(min_dist1) + pt.mean(min_dist2)

    del(pc1_expanded,pc2_expanded,dists,min_dist1,min_dist2)

    return chamfer_dist.item()




if __name__ == "__main__":
    argpsrMjr = getMjrArgPrsr()
    argpsrRgnclp = default_argument_parser()
    argpsrUlip = ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])

    tplRgnclpKarg = argpsrRgnclp.parse_known_args()
    tplUlipKarg = argpsrUlip.parse_known_args()
    tplMjrKarg = argpsrMjr.parse_known_args()

    nmspArgsRgnclp = tplRgnclpKarg[0]
    nmspArgsUlip = tplUlipKarg[0]
    nmspArgsMjr = tplMjrKarg[0]

    main()