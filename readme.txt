ICCV AIM 2019 Extreme Super-Resolution Challenge - Track 1: Fidelity
# http://www.vision.ee.ethz.ch/aim19/

#南京航空航天大学自动化学院404实验室 & 中国电子科技网络信息安全有限公司

#训练/测试代码见 EDSR https://github.com/thstkdgus35/EDSR-PyTorch
#参考 EDSR WDSR RDN AWSRN ESPCN
#非常感谢上述作者的开源，引用本文请引用他们.
#论文地址：https://arxiv.xilesou.top/abs/1912.08002
#使用测试代码为ESRGAN提供并修改，感谢分享
#若其他人测试与论文中出现值有较大出入（manga109我感觉挺高.不知道是不是数据集制作的问题.若有大佬验证请留言.）。请联系留言或邮箱：xxh96@outlook.com。
#若有相差不大（小于0.02），可能是我Best和最终模型弄混了。

#模型地址: 百度云链接: https://pan.baidu.com/s/1eC3yzVvoDeVbVITctZOmxQ 提取码: yvqk



测试方法：
cd ../../src
#[-]中为自己填写
#测试前先创建文件夹，并放模型 ../experiment/DSADCSR/model/dsadcsr_x16.pt
#若想自己训练需先训练SKIP再载入模型（论文描述）


#DSADCSR16 挑战赛模型
python main.py --model ADC16_12 --data_test Demo --n_resblock 4 --n_feats 128 --block_feat 384 --save DSADCSR --dir_data [/dataset/]  --scale 16  --pre_train ../experiment/DSADCSR/model/dsadcsr_x16.pt --test_only --save_result --prec half --self_ens

#ADCSR (ADCSRS --n_feats 64 --block_feat 192)
python main.py --model ADCSR --data_test [Demo+Set5+Set4] --n_resblock 4 --n_feats 128 --block_feat 384  --save [ADCSR_X2X3X4X8] --dir_data [/dataset/]  --scale [2 3 4 8]  --pre_train ../experiment/[--save]/model/[model_name] --test_only --save_result --prec half --self_ens

