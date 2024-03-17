# Isomorphic mesh generator (iMG)

The reference implementation for ["Isomorphic mesh generation from point clouds with multilayer perceptrons", IEEE Trans Vis Comput Graph. 2024.2](https://ieeexplore.ieee.org/document/10440526).

# Environment

- fml=0.1.0
- numpy=1.18.5
- point_cloud_utils=0.13.0
- pytorch=1.5.1
- torchvision=0.6.1
- open3d==0.10.0.1
- scikit-learn==0.23.1
- scipy==1.5.0

# Running the Code

0. Set a reference mesh (sphere.obj) and its anchor file (anchor32.txt) under "inputfile/common"
1. Place an input point cloud (***.obj) under "inputfile/[Input folder name]/[Object name]"
2. Run "main.py"
```
python main.py main.py [Input folder name] [Input file name] [Output folder name] [Scale factor \beta_1] [Num of epochs of global mapping] [Scale factor \beta_2] [Num of epochs of coarse local mapping] [Threshold \tau_e] [Scale factor \beta_3] [Num of epochs of fine local mapping] [Weight coefficient \alpha] [Noise flag] [Device]
```
3. Check final result "fine_local_mapping_result_e{}_s{}_global.obj"

# Demo

