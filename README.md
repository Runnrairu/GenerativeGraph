# Pytorch Implementation for Learning Deep Generative Models of Graphs
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, Peter Battaglia (DeepMind,London,UK)    
https://arxiv.org/abs/1803.03324  

## Usage

### Enviroments (CPU/GPU)
Pull the Envrioment for this implementation.  
`docker pull relutropy/research`  

Run the enviroment.  
CPU  
`docker run -i -t -p 1111:1111 relutropy/research /bin/bash`
GPU  
`nvidia-docker run -i -t -p 1111:1111 relutropy/research /bin/bash`  

Start jupyter server by below command.  
`In container $ nohup jupyter notebook --allow-root >> jupyter.log 2>&1 &`  

### Codes on Jupyter Notebooks
`git clone https://github.com/shllln/GenerativeGraph.git`  


#### Welcome PR and issue.
##### This is prtotype.  
There are some differences from the paper.  
For example, h_v_init has no condition vector and Datasets is Tox21 now.  
That will match the paper soon.  

