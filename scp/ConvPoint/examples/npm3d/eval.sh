
#!/bin/bash
echo "eval npm3d_seg.py..."
/home/s/anaconda3/envs/torch/bin/python eval.py \
--num_classes 20 \
--filename /media/s/TOSHIBA/dataset-semantic-seg/scp/train_pointclouds_downsample/whole_part1_10.npy \
--test \
--savedir /media/s/TOSHIBA/dataset-semantic-seg/scp_result/SegBig_8192_nocolorFalse_drop0.5_2022-07-27-08-47-23 \
--block_size 8

echo "done eval npm3d_seg.py..."
