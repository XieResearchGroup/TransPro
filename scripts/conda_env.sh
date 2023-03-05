# this is an example for cuda 11.3 
# change the version according to the pytorch and PYG documents. 
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install torch-geometric 
conda install -c conda-forge rdkit -y
pip install transformers 
pip install wandb 
