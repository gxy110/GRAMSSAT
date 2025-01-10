conda create -n gramssat python=3.8
conda activate gramssat
conda install cudatoolkit=11.0
python -m pip install opencv-python
python -m pip install transformers==3.4.0
conda install imbalanced-learn
conda install seaborn
conda install numpy
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install matplotlib
conda install dill
conda install scikit-learn
conda install pandas