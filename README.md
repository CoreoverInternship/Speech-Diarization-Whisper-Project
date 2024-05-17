You need to download CONDA, and CUDA to get this working. 
With both downloaded and the in the 'path' environmeent variable, create a env using the following command 'conda create --name <env_name> python=3.10'. 
Theen activate the env: 'conda activate <env_name>'.
Download important packages using 'conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia',
Download other required packages from requirements.txt using 'pip install -r requiremnts.txt'
