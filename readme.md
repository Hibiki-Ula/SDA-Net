# SDA-Net: Global Feature Point Cloud Completion Network with Serialization and Dual Attention


##  Information
- **Title**: SDA-Net: A global feature point cloud completion network based on serialization and dual attention
- **Journal**: IEEE Transactions on Visualization and Computer Graphics (TVCG)
- **Author**: [Wu weichao], [Xu yongyang], [Xie zhong]
- **Link**: comming

## 🛠️ Building

### Requirements
```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

# extension
```
1. cd into extensions/Module (eg extensions/gridding)
2. run `python setup.py install`
```
# PointNet++
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
# GPU kNN
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Dataset
Please refer to [Pointr](https://github.com/yuxumin/PoinTr)

### Pretrain_model
comming

### Usage
Please refer to the below
*TRAIN:*
```
bash scripts/train.sh 0 --config cfgs/PCN_models/GlobalFea.yaml --exp_name BaseLine
```

*TEST:*
```
bash scripts/test.sh 0 --ckpts model/GlobalFea/ShapeNet34/ckpt-best.pth --config cfgs/ShapeNet34_models/GlobalFea.yaml --mode easy --exp_name TestBaseLine
```

*INFERENCE::*
```
python tools/inference.py cfgs/ShapeNet55_models/GlobalFea.yaml model/GlobalFea/ShapeNet55/ckpt-best.pth --pc_root demo/demo_55/partial --out_pc_root demo/demo_55/result
```

KITTI:
```
python tools/inference.py cfgs/KITTI_models/GlobalFea.yaml model/GlobalFea/Kitti/ckpt-best.pth --pc_root demo/demo_kitti/partial --out_pc_root demo/demo_kitti/result
```
## License
MIT License

## Acknowledgements
Our code is inspired by [Pointr](https://github.com/yuxumin/PoinTr)


## Citation
If you find our work useful in your research, please consider citing: 
```
comming
```


