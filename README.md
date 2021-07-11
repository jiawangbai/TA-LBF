# TA-LBF
This repository provides the implementatin of our ICLR 2021 work: [Targeted Attack against Deep Neural Networks via Flipping Limited Weight Bits](https://openreview.net/forum?id=iKQAk8a2kM0).

## Abstract 
To explore the vulnerability of deep neural networks (DNNs), many attack paradigms have been well studied, such as the poisoning-based backdoor attack in the training stage and the adversarial attack in the inference stage. In this paper, we study a novel attack paradigm, which modifies model parameters in the deployment stage for malicious purposes. Specifically, our goal is to misclassify a specific sample into a target class without any sample modification, while not significantly reduce the prediction accuracy of other samples to ensure the stealthiness. To this end, we formulate this problem as a binary integer programming (BIP), since the parameters are stored as binary bits (*i.e.*, 0 and 1) in the memory. By utilizing the latest technique in integer programming, we equivalently reformulate this BIP problem as a continuous optimization problem, which can be effectively and efficiently solved using the alternating direction method of multipliers (ADMM) method. Consequently, the flipped critical bits can be easily determined through optimization, rather than using a heuristic strategy. Extensive experiments demonstrate the superiority of our method in attacking DNNs.

&nbsp;
&nbsp;
<div align=center>
<img src="https://github.com/jiawangbai/TA-LBF/blob/main/misc/demo.png" width="700" height="300" alt="Demonstration of TA-LBF"/><br/>
</div>
&nbsp;
&nbsp;

## Install 
1. Install PyTorch >= 1.5
2. Clone this repo:
```shell
git clone https://github.com/jiawangbai/TA-LBF.git
```

## Quick Start
Set the "cifar_root" in the "config.py" firstly.

Running the below command will attack a sample (3676-th sample in the CIFAR-10 validation set) into class 0.
```shell
python attack_one.py --target-class 0 --attack-idx 3676 --lam 100 --k 5
```
You can set "target-class" and "attack-idx" to perform TA-LBF on a specific sample.

## Reproduce Our Results
Set the "cifar_root" in the "config.py" firstly.

Running the below command can reproduce our results in attacking the 8-bit quantized ResNet on CIFAR-10 with the parameter searching strategy introduced in the paper.
```shell
python attack_reproduce.py 
```
"cifar_attack_info.txt" includes the 1,000 attacked samples and their target classes used in our experiments.
<br/>
Format:
<br/>
&emsp; [[target-class, sample-index],
<br/>
&emsp; [target-class, sample-index],
<br/>
&emsp; ...
<br/>
&emsp; [target-class, sample-index] ]
<br/>
where "sample-index" is the index of this attacked sample in CIFAR-10 validation set.

## Others
We provide the pretrained 8-bit quantized ResNet on CIFAR-10. -> "cifar_resnet_quan_8/model.th"

Python version is 3.6.10 and the main requirments are below:
<br/>
&emsp; torch==1.5.0
<br/>
&emsp; bitstring==3.1.7
<br/>
&emsp; torchvision==0.6.0a0+82fd1c8
<br/>
&emsp; numpy==1.18.1

We also provide the following command to install dependencies before running the code:
```shell
pip install -r requirements.txt
```

## Citation
```
@inproceedings{bai2021targeted,
  title={Targeted Attack against Deep Neural Networks via Flipping Limited Weight Bits},
  author={Bai, Jiawang and Wu, Baoyuan and Zhang, Yong and Li, Yiming and Li, Zhifeng and Xia, Shu-Tao},
  booktitle={ICLR},
  year={2021}
}
```
