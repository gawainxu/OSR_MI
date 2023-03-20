python3 train_ica.py --lr 1e-5 --epochs 50 --bz 128 --dim [128, 50, 30, 1] \
--feature_path "../features/cifar-10-10_vanillia_classifier32_0"  --feature_name "module.avgpool" \
--save_path "./save/cifar-10-10_vanillia_classifier32_0_module.avgpool_ica.pth"