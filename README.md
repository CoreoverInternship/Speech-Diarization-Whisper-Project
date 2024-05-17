You need to download CONDA(https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and CUDA(https://developer.nvidia.com/cuda-downloads) to get this working. 
Make sure both downloads are in the 'path' environment variable.
Create an env: 'conda create --name <env_name> python=3.10'. 
Activate the env: 'conda activate <env_name>'.
Download important packages using 'conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia',
Download other required packages from requirements.txt using 'pip install -r requiremnts.txt'
