import importlib
import torch

def loadModels(args):
    coarseModel = None
    fineModel = None
    detailModel = None
    if args.coarseModel:
        coarseModule = importlib.import_module(args.coarseModel)
        coarseModel = coarseModule.HourglassNet(args.coarse_nChannel)
        if args.coarseModel_load:
            coarseModel.load_state_dict(torch.load(args.coarseModel_load))
    
    if args.fineModel:
        fineModule = importlib.import_module(args.fineModel)
        fineModel = fineModule.DRN(args.fine_albedoChannel, 
                args.fine_normalChannel, args.find_lightingChannel)
        if args.fineModel_load:
            fineModel.load_state_dict(torch.load(args.fineModel_load))
    
    if args.detailModel:
        detailModule = importlib.import_module(args.detailModel)
        detailModel = detailModule.DRN(args.detail_albedoChannel,
                args.detail_normalChannel, args.detail_lightingChannel)
        if args.detailModel_load:
            detailModel.load_state_dict(torch.load(args.detailModel_load))
    return coarseModel, fineModel, detailModel
