ICCV AIM 2019 Extreme Super-Resolution Challenge - Track 1: Fidelity
# http://www.vision.ee.ethz.ch/aim19/

#南京航空航天大学自动化学院404实验室 & 中国电子科技网络信息安全有限公司

#训练/测试代码见 EDSR https://github.com/thstkdgus35/EDSR-PyTorch
#参考 EDSR WDSR RDN AWSRN ESPCN
#非常感谢上述作者的开源，引用本文请引用他们.
#论文地址：https://arxiv.xilesou.top/abs/1912.08002
#模型地址: 链接: https://pan.baidu.com/s/1eC3yzVvoDeVbVITctZOmxQ 提取码: yvqk



测试方法：
cd ../../src
python main.py --model ADC16_12 --data_test Demo --n_resblock 4 --n_feats 128 --block_feat 384  --save DSADCSR --dir_data [/dataset/]  --scale 16  --pre_train ../experiment/DSADCSR/model/model_best.pt --test_only --save_result --prec half --self_ens

