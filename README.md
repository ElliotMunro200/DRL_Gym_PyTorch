# DRL_Gym_PyTorch
DRL PyTorch implementations on OpenAI Gym environments.

1. Install conda and git.
2. Fork and/or clone the Git repository onto your local machine.
3. Install mujoco with the instructions in the Readme.md at https://github.com/openai/mujoco-py.
4. Add lines to your ~/.bashrc script:
   1. For making mujoco accessible: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/munel752/.mujoco/mujoco210/bin 
   2. For making nvidia accessible [optional]: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   3. For making libGLEW accessible [situational], lines like: export LD_PRELOAD=/usr/lib64/libGLEW.so.2.2
5. Create a conda environment for the repo with the RLRL.yml provided, by executing "conda env create -f RLRL.yml". This will create a conda env with the name "RLRL".

