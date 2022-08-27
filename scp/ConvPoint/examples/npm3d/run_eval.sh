
#!/bin/bash
echo "run npm3d_seg.py..."
/home/s/anaconda3/envs/torch/bin/python npm3d_seg_test.py \
--rootdir /media/s/TOSHIBA/dataset-semantic-seg/scp \
--savedir /media/s/TOSHIBA/dataset-semantic-seg/scp_result \
--num_classes 20 \
--ignore_index 20 \
--batch_size 1 \
--threads 4 \
--test

echo "done npm3d_seg.py..."
