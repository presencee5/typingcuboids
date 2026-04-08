from numba.np.ufunc import parallel
import open3d as o3d
import numpy as np
import torch
import pickle as pkl
import networkx as nx
import pytorch3d.renderer as pt3drdr
import pytorch3d.structures as pt3dstrc
import pickle as pkl
import numba as nb
import cupy
from itertools import combinations
from argparse import ArgumentParser
from os import makedirs
from tqdm import tqdm
from pymeshlab import MeshSet, Mesh
from matplotlib import pyplot as plt
from PIL import Image




def sample_points_on_cube_faces(cube_vertices, n_samples_per_edge=10):
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

        u = np.linspace(0, 1, n_samples_per_edge)
        v = np.linspace(0, 1, n_samples_per_edge)
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

    return np.vstack(sampled_points)



def find_closest_cube_index(query_points, list_of_cubes, device, n_samples_per_edge=10, bs="full"):
    query_points = torch.tensor(query_points, dtype=torch.float32, device=device)  # (N, 3)
    num_cubes = len(list_of_cubes) // 8
    cube_vertices = np.array(list_of_cubes).reshape((num_cubes, 8, 3))

    all_sampled_points = torch.zeros((num_cubes*n_samples_per_edge**2*6,3), device=device)
    scaI = 0
    for cube in cube_vertices:
        samples = sample_points_on_cube_faces(cube, n_samples_per_edge)
        all_sampled_points[scaI:scaI+n_samples_per_edge**2*6] = torch.tensor(samples, dtype=torch.float32, device=device)
        scaI += n_samples_per_edge**2*6
    all_sampled_points = torch.unsqueeze(all_sampled_points,dim=0)

    closest_indices = []
    query_points = query_points.unsqueeze(1)  # (N, 1, 3)

    batchsize = None
    if bs == "full":
        batchsize = query_points.shape[0]
    else:
        batchsize = int(bs)
    for intI in range(0, query_points.shape[0], batchsize):
        bchQryPnts = query_points[intI:intI+batchsize].expand((batchsize,all_sampled_points.size(1),3))
        bchDists = torch.sum((bchQryPnts-all_sampled_points)**2,2)
        bchIndices = torch.argmin(bchDists, dim=1) // (n_samples_per_edge**2*6)  # (ll)
        closest_indices.extend(bchIndices.detach().cpu().numpy())

    return np.array(closest_indices)



def build_index_graph_adjmtx(vertices, faces):
    vertex_to_index = vertices[:,3].astype(int)

    num_indices = max(vertex_to_index) + 1
    graph = np.zeros((num_indices, num_indices), dtype=bool)

    for face in faces:
        idxs = vertex_to_index[face]
        if idxs[0] != idxs[1]:
            graph[idxs[0],idxs[1]] = True
            graph[idxs[1],idxs[0]] = True
        if idxs[0] != idxs[2]:
            graph[idxs[0],idxs[2]] = True
            graph[idxs[2],idxs[0]] = True
        if idxs[1] != idxs[2]:
            graph[idxs[1],idxs[2]] = True
            graph[idxs[2],idxs[1]] = True

    return graph



def get_extrinsic(rx, ry, d):
    aryRotX = np.eye(3,dtype=np.float32)
    aryRotY = np.eye(3,dtype=np.float32)
    trans = np.eye(4,dtype=np.float32)
    aryRotX[1,1] = np.cos(rx)
    aryRotX[1,2] = -np.sin(rx)
    aryRotX[2,1] = np.sin(rx)
    aryRotX[2,2] = np.cos(rx)
    aryRotY[0,0] = np.cos(ry)
    aryRotY[0,2] = np.sin(ry)
    aryRotY[2,0] = -np.sin(ry)
    aryRotY[2,2] = np.cos(ry)
    trans[:3,:3] = np.dot(aryRotY,aryRotX)
    trans[2,3] = d
    return trans



def preprocess(model, mdltyp):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    if mdltyp == "msh":
        vertices = np.asarray(model.vertices)
        vertices -= center
        model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    else:
        vertices = np.asarray(model.points)
        vertices -= center
        model.points = o3d.utility.Vector3dVector(vertices / scale)
    return (model, center, scale)



def voxel_carving(mesh, cubic_size, voxel_resolution, vws, w=300, h=300):
    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # rescale geometry
    tplPrepRslt = preprocess(mesh, "pnt")
    mesh = tplPrepRslt[0]

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    scaI = 0
    for scaI in range(vws.shape[0]):
        # get new camera pose
        trans = get_extrinsic(vws[scaI,0],vws[scaI,1],3)
        param.extrinsic = trans
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)

        # depth map carving method
        voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
    vis.destroy_window()

    return (voxel_carving, mesh, tplPrepRslt[1], tplPrepRslt[2])



def voxgIsosfExtrc(vxgrd):
    # usage in `main()`:
    #     o3dmsh = voxgIsosfExtrc(o3dvxg)
    #     o3d.io.write_triangle_mesh("rsltMsh.ply",o3dmsh)
    lstTrgtVx = vxgrd.get_voxels()
    aryTrgtVx = np.zeros((len(lstTrgtVx), 3), dtype=int)
    scaI = 0
    while scaI < len(lstTrgtVx):
        aryTrgtVx[scaI] = lstTrgtVx[scaI].grid_index
        scaI += 1

    aryTrgtVx = aryTrgtVx[:, [2, 1, 0]]

    aryFld = np.zeros(np.max(aryTrgtVx, axis=0)+3)
    scaI = 0
    while scaI < aryTrgtVx.shape[0]:
        aryFld[1+aryTrgtVx[scaI, 0], 1 +
               aryTrgtVx[scaI, 1], 1+aryTrgtVx[scaI, 2]] = 1
        scaI += 1

    o3dmshRslt = o3d.t.geometry.TriangleMesh.create_isosurfaces(
        o3d.core.Tensor(aryFld), [0.5]).to_legacy()

    aryTmp = np.asarray(o3dmshRslt.vertices)
    aryTmp -= 1
    aryTmp = (aryTmp - (o3dmshRslt.get_max_bound() +
              o3dmshRslt.get_min_bound())/2) * vxgrd.voxel_size
    o3dmshRslt.vertices = o3d.utility.Vector3dVector(aryTmp)

    return o3dmshRslt



def denVoxCreate(pnt,cubicsz=2.0,reslut=128):
    scaI = 0
    scaJ = 0
    lstVws = []
    for scaI in range(0,360,36):
        for scaJ in range(0,360,36):
            lstVws.append([scaI,scaJ])
    aryVws = np.array(lstVws,dtype=np.float32)/360*2*np.pi

    return voxel_carving(pnt, cubicsz, reslut, aryVws)



def rasandrdr(pnt, dvc,
              azmele=[[50, -35],[50, 55],[50, 145],[50, 235],
                      [35, -35],[35, 10],[35, 55],[35, 100],[35, 145],[35, 190],[35, 235],[35, 280],
                      [-10, -35],[-10, 55],[-10, 145],[-10, 235],
                      [-55, -35],[-55, 10],[-55, 55],[-55, 100],[-55, 145],[-55, 190],[-55, 235],[-55, 280]],
              distofst=0.9):
    '''
    [[0,0],[0,60],[0,120],[0,180],[0,240],[0,300],
     [60,0],[60,60],[60,120],[60,180],[60,240],[60,300],
     [120,0],[120,60],[120,120],[120,180],[120,240],[120,300],
     [180,0],[180,60],[180,120],[180,180],[180,240],[180,300],
     [240,0],[240,60],[240,120],[240,180],[240,240],[240,300],
     [300,0],[300,60],[300,120],[300,180],[300,240],[300,300]]
    '''
    torch.set_grad_enabled(False)
    tsrPnt = torch.from_numpy(pnt).to(torch.float32)
    lstAzm = np.array(azmele)[:,0].tolist()
    lstEle = np.array(azmele)[:,1].tolist()
    tsrMaxBnd = torch.max(tsrPnt[:,0:3],dim=0)[0]
    tsrMinBnd = torch.min(tsrPnt[:,0:3],dim=0)[0]
    tsrPnt[:,0:3] = tsrPnt[:,0:3] - (tsrMinBnd+(tsrMaxBnd-tsrMinBnd)/2)
    tsrPnt[:,0:3] = tsrPnt[:,0:3] / (torch.linalg.vector_norm(tsrMaxBnd-tsrMinBnd)/2).item()
    tsrMaxBnd = torch.max(tsrPnt[:,0:3],dim=0)[0]
    tsrMinBnd = torch.min(tsrPnt[:,0:3],dim=0)[0]
    scaCamDist = torch.linalg.vector_norm(tsrMaxBnd-tsrMinBnd)/2 + distofst
    pt3dpnt = pt3dstrc.Pointclouds(points=torch.unsqueeze(tsrPnt[:,0:3],0).expand(len(lstAzm),tsrPnt.size(0),3).to(dvc),
                                   features=torch.unsqueeze(tsrPnt[:,3:6],0).expand(len(lstAzm),tsrPnt.size(0),3).to(dvc))

    tplCamRt = pt3drdr.look_at_view_transform(scaCamDist,lstAzm,lstEle)
    pt3dcam = pt3drdr.FoVPerspectiveCameras(znear=0.01, R=tplCamRt[0], T=tplCamRt[1], device=dvc)
    pt3drasst = pt3drdr.PointsRasterizationSettings(image_size=800, radius=0.01, points_per_pixel=1, max_points_per_bin=70000)
    pt3dras = pt3drdr.PointsRasterizer(cameras=pt3dcam, raster_settings=pt3drasst)
    pt3dcmpst = pt3drdr.NormWeightedCompositor(background_color=(240,240,240))
    pt3drdrr = pt3drdr.PointsRenderer(rasterizer=pt3dras,compositor=pt3dcmpst)
    tsrIdxs= pt3dras(pt3dpnt).idx
    tsrImgs = torch.round(pt3drdrr(pt3dpnt))
    tsrIdxs[tsrIdxs!=-1] = tsrIdxs[tsrIdxs!=-1] % 65536
    torch.set_grad_enabled(True)

    # scaI = 0
    # aryTmp = tsrImgs.cpu().detach().numpy().astype(np.uint8)
    # for scaI in range(36):
    #     plt.imsave(str(scaI)+".png",aryTmp[scaI])

    return (tsrIdxs.cpu().detach().numpy(),
            tsrImgs.cpu().detach().numpy().astype(np.uint8))



@nb.njit(parallel=True)
def numbaisin(ele,tstele):
    a = np.reshape(ele,(ele.shape[0]*ele.shape[1]))
    out=np.empty(a.shape[0], dtype=nb.boolean)
    b = set(tstele)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return np.reshape(out,ele.shape)



def ehcgppnttobx(pntid,idximg,nbxthrs=10):
    # Image.fromarray(np.squeeze(np.round((idximg!=-1).astype(np.uint8))*255)).save("idx.png")
    aryMsk = np.squeeze(numbaisin(idximg.astype(np.int32),pntid.astype(np.int32)))
    # aryMsk = np.round(aryMsk.astype(int)).astype(np.uint8)*255
    # Image.fromarray(np.squeeze(aryMsk)).save("intersect.png")
    if np.sum(aryMsk) < nbxthrs:
        return None

    aryRslt = np.zeros((1,5),dtype=np.uint16)
    aryMskRgn = np.argwhere(aryMsk)
    aryRslt[0,[0,2]] = np.min(aryMskRgn,axis=0).astype(np.uint16)
    aryRslt[0,[1,3]] = np.max(aryMskRgn,axis=0).astype(np.uint16)
    return aryRslt



def ehcgppnttobxcupy(pntid,idximg,nbxthrs=10):
    # Image.fromarray(np.squeeze(np.round((idximg!=-1).astype(np.uint8))*255)).save("idx.png")
    aryMsk = cupy.asnumpy(cupy.squeeze(cupy.isin(idximg,cupy.array(pntid.astype(np.int32)))))
    # aryMsk = np.round(aryMsk.astype(int)).astype(np.uint8)*255
    # Image.fromarray(np.squeeze(aryMsk)).save("intersect.png")
    if np.sum(aryMsk) < nbxthrs:
        return None

    aryRslt = np.zeros((1,5),dtype=np.uint16)
    aryMskRgn = np.argwhere(aryMsk)
    aryRslt[0,[0,2]] = np.min(aryMskRgn,axis=0).astype(np.uint16)
    aryRslt[0,[1,3]] = np.max(aryMskRgn,axis=0).astype(np.uint16)
    return aryRslt



def grpcmb(cmb,gph):
    tplSubgIdx = np.meshgrid(cmb, cmb)
    arySubGph = gph[tplSubgIdx[0],tplSubgIdx[1]].astype(int)
    nxgph = nx.from_numpy_array(arySubGph)
    lstCntCom = list(nx.connected_components(nxgph))
    scaI = 0
    for scaI in range(len(lstCntCom)):
        lstCntCom[scaI] = np.array(cmb,dtype=int)[np.array(list(lstCntCom[scaI]),dtype=int)].tolist()
    return lstCntCom



def cmbgrptopntid(cmbgrp,ttlpnt):
    aryRslt = np.nonzero(ttlpnt[:,6]==cmbgrp[0])[0]
    scaI = 0
    for scaI in range(1,len(cmbgrp)):
        aryRslt = np.concatenate((aryRslt,np.nonzero(ttlpnt[:,6]==cmbgrp[scaI])[0]),axis=0)
    return aryRslt



def drwrectang(img, x0, y0, x1, y1):
    color = np.array([255,0,0])
    img = img.astype(np.float64)
    img[y0:y1, x0-1:x0+2, :3] = color
    img[y0:y1, x1-1:x1+2, :3] = color
    img[y0-1:y0+2, x0:x1, :3] = color
    img[y1-1:y1+2, x0:x1, :3] = color
    # img[y0:y1, x0:x1, :3] /= 2
    # img[y0:y1, x0:x1, :3] += color * 0.5
    img = img.astype(np.uint8)
    return img



def sveapairforvis(svepth,cubcmbvtcs,boxset,imgset,npntsmpcubpe=10):
    aryCubSmp = np.zeros((cubcmbvtcs.shape[0],6*npntsmpcubpe**2,3))
    scaI = 0
    for scaI in range(cubcmbvtcs.shape[0]):
        aryCubSmp[scaI] = sample_points_on_cube_faces(cubcmbvtcs[scaI],npntsmpcubpe)
    np.savetxt(svepth+".xyz",np.reshape(aryCubSmp,[aryCubSmp.shape[0]*aryCubSmp.shape[1],3]))
    for scaI in range(boxset.shape[0]):
        Image.fromarray(drwrectang(imgset[boxset[scaI,4]],
                                   boxset[scaI,2],
                                   boxset[scaI,0],
                                   boxset[scaI,3],
                                   boxset[scaI,1])).save(svepth+"Bx"+str(scaI)+".png")



def bldgph(inpntdir,npnts,incubdir,outdir,device="cuda:0"):
    filTmp = open(inpntdir+"objId.txt")
    lstInputNme = filTmp.readlines()
    filTmp.close()
    ptdvc = torch.device(device)
    scaI = 0
    aryTmp = None
    tplRslt = None
    dctRslt = {}

    for scaI in tqdm(range(len(lstInputNme)),ncols=50):
        aryTmp = np.load(inpntdir+lstInputNme[scaI][0:-1]+"Pts"+npnts+".npy")
        o3dpnt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate((np.expand_dims(aryTmp["x"],1),
                                                                                    np.expand_dims(aryTmp["y"],1),
                                                                                    np.expand_dims(aryTmp["z"],1)),axis=1)))
        o3dmshCub = o3d.io.read_triangle_mesh(incubdir+lstInputNme[scaI][0:-1]+"_cube_masked.ply")
        tplRslt = denVoxCreate(o3dpnt)
        # o3d.visualization.draw_geometries(tplRslt[0])

        od3mshVx = voxgIsosfExtrc(tplRslt[0])
        o3dmshCub.vertices = o3d.utility.Vector3dVector((np.asarray(o3dmshCub.vertices) - tplRslt[2]) / tplRslt[3])

        aryNearIdx = find_closest_cube_index(np.asarray(od3mshVx.vertices),
                                             np.asarray(o3dmshCub.vertices),
                                             device=ptdvc)
        '''
        pmslms = MeshSet()
        pmslms.add_mesh(Mesh(vertex_matrix=np.asarray(od3mshVx.vertices),face_matrix=np.asarray(od3mshVx.triangles),
                             v_quality_array=np.expand_dims(aryNearIdx,axis=1)))
        pmslms.save_current_mesh("temp.ply")
        exit()
        '''

        dctRslt[lstInputNme[scaI][0:-1]] = build_index_graph_adjmtx(np.hstack((np.asarray(od3mshVx.vertices), aryNearIdx.reshape(-1, 1))),
                                                                    np.asarray(od3mshVx.triangles))
    filTmp = open(outdir+"gphs.pkl","wb")
    pkl.dump(dctRslt,filTmp)
    filTmp.close()



def bldpair(inpntdir,npnts,incubdir,seleidpth,outdir,device="cuda:0"):
    filTmp = open(inpntdir+"objId.txt")
    lstInputNme = filTmp.readlines()
    filTmp.close()
    filTmp = open(outdir+"gphs.pkl","rb")
    dctGph = pkl.load(filTmp)
    filTmp.close()
    arySeleId = np.load(seleidpth).astype(int)
    ptdvc = torch.device(device)
    scaI = 0
    aryPnt = None
    aryGph = None
    tplMvrdr = None
    aryCubs = None
    lstCmbsId = []
    lstCmbGrp = None
    lstEchCgrp = None
    aryEhcgpPnt = None
    scaJ = 0
    scaK = 0
    aryBxes = None
    scaBxesLen = 0
    aryBx = None
    lstCmbBxs = []
    dctCmbBx = {}

    for scaI in range(len(lstInputNme)):
        if not np.isin(scaI,arySeleId):
            print("{}/{}\nDropped.".format(scaI,len(lstInputNme)-1))
            continue
        print("{}/{}".format(scaI,len(lstInputNme)-1))
        torch.cuda.empty_cache()
        aryPnt = np.load(inpntdir+lstInputNme[scaI][0:-1]+"Pts"+npnts+".npy")
        aryCubs = np.asarray(o3d.io.read_triangle_mesh(incubdir+lstInputNme[scaI][0:-1]+"_cube_masked.ply").vertices).astype(np.float32)
        aryGph = dctGph[lstInputNme[scaI][0:-1]]

        aryPnt = np.concatenate((np.expand_dims(aryPnt["x"],1),
                                 np.expand_dims(aryPnt["y"],1),
                                 np.expand_dims(aryPnt["z"],1),
                                 np.expand_dims(aryPnt["r"],1),
                                 np.expand_dims(aryPnt["g"],1),
                                 np.expand_dims(aryPnt["b"],1)),axis=1)
        aryPnt = np.concatenate((aryPnt,np.expand_dims(find_closest_cube_index(aryPnt[:,0:3],aryCubs,device=ptdvc),1)),axis=1)

        aryCubs = np.reshape(aryCubs,(aryCubs.shape[0]//8,8,3))

        tplMvrdr = rasandrdr(aryPnt,ptdvc)

        lstCmbsId = []
        for scaJ in range(1,aryCubs.shape[0]):
            lstCmbsId.extend([np.array(list(lstTmp)).astype(np.uint8) for lstTmp in combinations(range(aryCubs.shape[0]),scaJ)])

        aryBxes = np.zeros((50*tplMvrdr[0].shape[0],5),dtype=np.uint16)
        lstCmbBxs = []
        for scaJ in tqdm(range(len(lstCmbsId)),ncols=50):
            dctCmbBx = {}
            dctCmbBx["cmbid"] = lstCmbsId[scaJ]

            lstCmbGrp = grpcmb(lstCmbsId[scaJ], aryGph)
            scaBxesLen = 0
            for lstEchCgrp in lstCmbGrp:
                aryEhcgpPnt = cmbgrptopntid(lstEchCgrp,aryPnt)
                for scaK in range(0,tplMvrdr[0].shape[0]):
                    aryBx = ehcgppnttobx(aryEhcgpPnt,tplMvrdr[0][scaK])
                    if aryBx is None:
                        continue
                    else:
                        aryBx[0,4] = scaK
                        if scaBxesLen >= aryBxes.shape[0]:
                            print("Out of `aryBoxes`.")
                            aryBxes = np.concatenate((aryBxes,
                                                      np.zeros((50*tplMvrdr[0].shape[0],5),dtype=np.uint16)),axis=0)
                        aryBxes[scaBxesLen] = aryBx
                        scaBxesLen += 1
            dctCmbBx["boxes"] = aryBxes[0:scaBxesLen]
            lstCmbBxs.append(dctCmbBx)
            '''
            if len(lstCmbGrp) > 1:
                sveapairforvis("./cmb"+str(scaJ)+"Obj"+lstInputNme[scaI][0:-1],
                               aryCubs[dctCmbBx["cmbid"]],
                               dctCmbBx["boxes"],
                               tplMvrdr[1])
                exit()
            '''
        np.save(outdir+lstInputNme[scaI][0:-1]+".npy",tplMvrdr[1])
        filTmp = open(outdir+lstInputNme[scaI][0:-1]+".pkl", "wb")
        pkl.dump({"cmbbxs":lstCmbBxs,"cubabs":aryCubs},filTmp)
        filTmp.close()



def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--strCtgid")
    parser.add_argument("-p", "--strInPnt", default="../dataset/output/")
    parser.add_argument("-n", "--strNpnts", default="65536")
    parser.add_argument("-u", "--strInCub")
    parser.add_argument("-s", "--strSeleIdPth")
    parser.add_argument("-f", "--strFunc")
    parser.add_argument("-o", "--strOutRoot", default="./output/")
    nmspArgs = parser.parse_args()

    if nmspArgs.strFunc == "bldgph":
        makedirs(nmspArgs.strOutRoot+nmspArgs.strCtgid,exist_ok=True)
        bldgph(nmspArgs.strInPnt+nmspArgs.strCtgid+"Final/train/",nmspArgs.strNpnts,nmspArgs.strInCub,nmspArgs.strOutRoot+nmspArgs.strCtgid+"/")
        print("Done!")
        exit(0)
    elif nmspArgs.strFunc == "bldpair":
        bldpair(nmspArgs.strInPnt+nmspArgs.strCtgid+"Final/train/",nmspArgs.strNpnts,nmspArgs.strInCub,nmspArgs.strSeleIdPth,
                nmspArgs.strOutRoot+nmspArgs.strCtgid+"/")
        print("Done!")
        exit(0)
    else:
        print("Invalid `strFunc`. Exiting...")
        exit(1)




if __name__ == "__main__":
    main()