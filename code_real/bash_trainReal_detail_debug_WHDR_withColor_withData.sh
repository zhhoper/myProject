python trainSUNCG_detail_real_debug_WHDR_withColor_withData.py --batchSize 8 --coarseModel 'defineHourglass_64' --fineModel 'defineDRN_normal' --detailModel 'defineDRN_normal' --coarseModel_load '../result/result_coarse_real_debug_WHDR_withColor_0.0010_0.00_0020/trained_model.t7' --fineModel_load '../result/result_fine_real_debug_WHDR_withColor_0.0010_0.00_0020/trained_model.t7' --detailModel_load '../result/continue_result_thirdStage_DRN_normal_0.0010_0.00_0020/trained_model.t7' --imageSize 256 --savePath '../result/result_detail_real_debug_WHDR_withColor_batch_8'