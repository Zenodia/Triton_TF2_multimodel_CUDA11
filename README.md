This repository an minimalistic experiment on Triton serving multiple models ( 2 to be exactly with T) using latest CUDA 11 and TensorRT7.X
with tensorflow 2.2 
## big thanks to Anas Abidin aabidin@nvidia.com , his triton and tensorRT deepdive webniar is brilliant, my experiment is a modification for Tensorflow 2.2 based on his work !

# get the data 
go to this [my google drive](https://drive.google.com/drive/folders/1zC0f3T-jdNiP_6mDO4AWxtyRn-YiX2GX?usp=sharing)
 to download the spiral hand writting data and place it under data folder


# Setting up the environment

below script will build and run a docker image with a data directory mounted and a choosen GPU( I use GPU 6 )
noted that it is using pytorch:20.06 as base image which requires CUDA 11.0 and driver 450 

```
./0a_startDocker.sh 6 <path_to_data_dir>
```
Once inside this container, please use the following script to enable GPU dashboards
```
./0b_installNVDashboardInDocker.sh
```

also install some python packages 
```
./0c_pip_install_packages.sh
```

once inside this container , launch jupyter notebook using the below script
```
./0d_start_jupyter.sh
```
then open a browser ( I use firefox) and then type in 0.0.0.0:23579 
 
## open another terminal to launch TRITON server  
A separate container for the Triton server needs to be launched using the script 

```
./1_start_trtis_server.sh 6 <full_path_to_an_empty_dir>
```


you should see something similar to the below screenshot 



![start_triton_server_successfully](<./pics/start_triton_server_successfully.JPG>) 

#### TRITON metrics
To launch Grafana dashboards for monitoring of metrics, please run `docker-compose up` from the [monitoring](./monitoring/) folder and navigate to [localhost:3000/](http://localhost:3000). Additional steps [here](./monitoring/readme.md).


# Notebooks

The three notebooks in this repository walkthrough the example steps for interactively deploying your model to Triton server and make inference



### Step_1. build and train the first Tensorflow CNN model, save the model for step 2




2a_parkinson_detection.ipynb 



### Step_2. convert the tensorflow saved model to model.onnx and model.plan, at the same time construct config.pbtxt and labels.txt files



2b_h52onnx2plan.ipynb



### Step_3. Triton inferencing interactively  step by step with this notebook below



3_triton_inference.ipynb  

(optional - you don't need to do this manually, it's covered in the notebook )
open yet another terminal, then recursively copy the 1st model to the empty_dir/ to serve your first model ( under the folder custom_plan/ ) 
```
cp -R model_repository/custom_plan/ empty_dir/custom_plan/
```
Note: you should verify at the Triton server terminal, the custom_plan model is loaded successfully

![Triton serves the custom plan model](<./pics/custom_plan_load_successfully.JPG>) 



recursively copy the custom_onnx to the empty_dir/ ( under the folder custom_onnx/) to serve your onnx model 
you should see something similar to the below in triton server updated interactively the newly populated model

```
cp -R model_repository/custom_onnx/ empty_dir/custom_onnx/
```

Note: you should verify at the Triton server terminal, the custom_onnx model is loaded successfully


![Triton serves the custom_onnx model](<./pics/custom_onnx_load_successfully.JPG>) 


# crediting

This experiment uses code and utilities from the following tools were used as part of this code base and are governed by their respective license agreements. These are in addition to tools distributed within the NGC Docker containers ([Pytorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) / [TRITON](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)).

* [MONAI](https://github.com/Project-MONAI/MONAI/blob/master/LICENSE)
* [Netron](https://github.com/lutzroeder/netron/blob/main/LICENSE)
* [NV Dashboard](https://github.com/rapidsai/jupyterlab-nvdashboard/blob/branch-0.4/LICENSE.txt)
