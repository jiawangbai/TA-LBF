
# path of cifar10 data on yout device
cifar_root = "../cifar10_data"


# path of the quantized model
model_root = "cifar_resnet_quan_8"


# The 1,000 attacked samples used in our experiments
# Format: [ [target-class, sample-index],
# 	      [target-class, sample-index],
# 	       ...
# 	      [target-class, sample-index] ]
info_root = "cifar_attack_info.txt"
