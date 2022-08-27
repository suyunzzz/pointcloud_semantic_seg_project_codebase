
#!/bin/bash
echo "run npm3d_seg.py..."
/home/s/anaconda3/envs/torch/bin/python npm3d_seg.py \
--rootdir /media/s/TOSHIBA/dataset-semantic-seg/Lille_preprocessed \
--savedir /media/s/TOSHIBA/dataset-semantic-seg/Lille_result

echo "done npm3d_seg.py..."
