# CUDA_VISIBLE_DEVICES=2 python train_ssd.py --dataset_type city_scapes --datasets data --validation_dataset data --base_net models/mb2-imagenet-71_8.pth
# python train_ssd.py --dataset_type city_scapes --datasets data --validation_dataset data --base_net models/mb2-imagenet-71_8.pth --net mb3-large-ssd-lite
# python continual_ssd.py --dataset_type city_scapes --datasets data --validation_dataset data --net mb2-ssd-lite --resume models/mb2-ssd-lite-mp-0_686.pth --use_cuda
CUDA_VISIBLE_DEVICES=0 python -u continual_ssd_tvmodel.py  | tee log/continual.log
CUDA_VISIBLE_DEVICES=1 python -u continual_ssd_tvmodel.py  --freeze_base_net | tee log/continual_freezebasenet.log
CUDA_VISIBLE_DEVICES=2 python -u continual_ssd_tvmodel.py  --freeze_net | tee log/continual_freezenet.log
