# CSI5388project
## Dependency:
* `tensorflow-gpu==2.2.0`
* `numpy==1.17.2`
* `matplotlib==3.1.1`
## To run the experiment:
* If you do not have victim model trained:
`python experiments.py -victim -attack`
* If you already have victim model weights stored in the folder:
`python experiments.py -attack`
* For helps:
`python experiments.py -h`
## File Structure:
* Training results can be found: `logs/`
* Weights can be found: `attack/`, `victim/`
## Nomination(For folder and csv names):
* `cifar`: using the dataset CIFAR10
* `mnist`: using the dataset MNIST
* `mixed`: using the combined loss，0.5 for victim model predicted labels, 0.5 for ground truth loss
`Loss = 0.5*sparse_category_entropy(Y_victim, Y_pred) + 0.5*sparse_category_entropy(Y_true, Y_pred)`
* `pretrain`: training the attack model from weights prtained on ImageNet
* `usePretrain`: using the victim which trained from weights prtained on ImageNet
