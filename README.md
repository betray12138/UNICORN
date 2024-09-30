## Towards an Information Theoretic Framework of Context-Based Offline Meta-Reinforcement Learning (UNICORN)

# 1. Reproducing Environment
    GPU: NVIDIA A100-SXM4-80GB
    CPU: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
    NVIDIA-SMI: 515.43.04
    CUDA Version: 11.7

    Install the MuJoCo according to [OpenAI guideline](https://github.com/openai/mujoco-py). To reproduce, you need to set the directory **.mujoco** under ~/. , and we provide the file-tree as follows:
        --.mujoco/
            --mjkey.txt
            --mjpro131/
            --mujoco210/

    Then do the following procedures to install unicorn.
        conda create --name unicorn python=3.10.4
        conda activate unicorn 
        pip install setuptools==59.5.0
        pip install wheel==0.37.1
        pip install cython==0.29.32
        pip install patchelf
        pip install pyOpenGL -i https://pypi.douban.com/simple
        pip install -r requirements.txt
        pip install -U 'mujoco-py<2.2,>=2.1'
        pip install gin-config
        pip install scikit-learn
        pip install seaborn==0.11.2
        pip install tensorboardX==2.6.2

    ** NOTE: To reproduce the experiment results, please strictly follow versions of torch, Python and CUDA!!! CUDA version is 11.7, Python version is 3.10.4, they are listed above. Torch version is 2.0.1 that can be found at requirements.txt

# 2. Prepare for the dataset
    As collecting the dataset need a huge amount of time, we provide two datasets to valid as an example. You can download the datasets like [env_name].tar.bz2 [At Anonymous Site Here](https://drive.google.com/drive/folders/1pCoot1fWSWqBlE64pAJcJkQTmoT4tqT7?usp=sharing), and then extract them under the directory called **batch_data**.
        --batch_data/
            --AntDir-v0/
                --data/
                    --seed_1_goal_2.62/
                        --obs.npy
                        --actions.npy
                        --next_obs.npy
                        --rewards.npy
                        --terminals.npy
                    --seed_2_goal_2.739/
                    ...
                    --seed_40_goal_2.562/

    To test the behaviour-ood performance, you need to download the datasets like [env_name]_model.tar.bz2 [At Anonymous Site Here](https://drive.google.com/drive/folders/1pCoot1fWSWqBlE64pAJcJkQTmoT4tqT7?usp=sharing), and then extract them under the directory called **batch_data_copy**.
        --batch_data_copy/
            --AntDir-v0/
                --data/
                    --seed_1_goal_2.62/
                        --models/
                            --agentxx.pt
                    --seed_2_goal_2.739/
                    ...
                    --seed_40_goal_2.562/

# 3. Run the code
    You can use the following command to run the code:
    conda activate unicorn
    python train_offline_FOCAL.py --env-type ant_dir(hopper_param)

# 4. Quick fix
    If there shows an error that module numpy has no attribute 'int' or module numpy has no attribute 'bool', please substitute np.bool to bool or np.int to int in the code. This is because numpy no longer supports np.int or np.bool after 1.20, some of our libraries are using an older version of numpy

# 5. Citation
    If you find the codebase is helpful for you, please cite 

    @article{li2024towards,
        title={Towards an Information Theoretic Framework of Context-Based Offline Meta-Reinforcement Learning},
        author={Li, Lanqing and Zhang, Hai and Zhang, Xinyu and Zhu, Shatong and Zhao, Junqiao and Heng, Pheng-Ann},
        journal={arXiv preprint arXiv:2402.02429},
        year={2024}
    }