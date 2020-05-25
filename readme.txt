ICCV AIM 2019 Extreme Super-Resolution Challenge - Track 1: Fidelity
# http://www.vision.ee.ethz.ch/aim19/

#南京航空航天大学自动化学院404实验室 & 中国电子科技网络信息安全有限公司

#训练/测试代码详见:EDSR https://github.com/thstkdgus35/EDSR-PyTorch
#参考 EDSR WDSR RDN AWSRN ESPCN
#非常感谢上述作者的开源，引用本文请引用他们.
#论文地址：https://arxiv.org/abs/1912.08002
#论文已被ICCVW2019接收,但由于延迟提交,在CVF中看不到.
#使用测试代码为ESRGAN提供并修改，感谢分享
#若其他人测试与论文中出现值有较大出入（我感觉manga109的测试结果挺高.不知道是不是LR数据集制作的问题.若有大佬验证请留言.）
#邮箱：xxh96@outlook.com。

#若有相差不大（小于0.02），可能是我Best和最终模型弄混了。
#代码中使用WDSR的WN作为归一化层，论文中没提及，也懒得改了，望见谅。
#再次感谢大佬们提供的代码。
#模型地址: 百度云链接: 链接: https://pan.baidu.com/s/1_TY-dR_J7RrsZNfVtPQHJA 提取码: b7in

测试方法：
cd ../../src

#src/option.py  set:
parser.add_argument('--act', type=str, default='relu',
           help='activation function') 
           
#src/option.py  add:
parser.add_argument('--alpha', type=float, default=1.0,
           help='alpha')
parser.add_argument('--beta', type=float, default=1.0,
           help='beta')
parser.add_argument('--block_feats', type=int, default=384,
            help='number of block_feats feature maps')
parser.add_argument('--r_mean', type=float, default=0.4488,
                    help='Mean of R Channel')
parser.add_argument('--g_mean', type=float, default=0.4371,
                    help='Mean of G channel')
parser.add_argument('--b_mean', type=float, default=0.4040,
                    help='Mean of B channel')
            
#[ ]中需决定自己填写内容，并删除'[]'.
#测试前先创建文件夹，并放模型 ../experiment/DSADCSR/model/dsadcsr_x16.pt
#若想自己训练需先训练SKIP再载入模型（如论文中描述）

#测试代码 半精度 + Self-Ensemble
#DSADCSR 16为挑战赛模型
python main.py --model DSADCSR --data_test [Demo+Set5] --n_resblock 4 --n_feats 128 --block_feat 384 --save [DSADCSR] --dir_data [/dataset/]  --scale [8 16]  --pre_train ../experiment/[DSADCSR]/model/[dsadcsr_x16.pt] --test_only --save_result --prec half --self_ens

#ADCSR (ADCSRS --n_feats 64 --block_feat 192)
python main.py --model ADCSR --data_test [Demo+Set5+Set4+...] --n_resblock 4 --n_feats 128 --block_feat 384  --save [ADCSR_X2X3X4X8] --dir_data [/dataset/]  --scale [2 3 4 8]  --pre_train ../experiment/[--save]/model/[model_name] --test_only --save_result --prec half --self_ens


#2020年2月23日更新
#修改了adcsr与dsadcsr的激活函数选择。
#为我的代码错误，按照原始方法不能设置act为leakyrelu
#因此使用修改后的代码需将act设置为relu，与论文中模型不符。若有机会我将会更正最终测试数据，但需要等待几个月。

#论文中adcsr的基础测试集数据的激活函数为relu！！！
#dsadcsr的第一级为relu，第二级由于是直接设置，激活函数为leakyrelu。
#带来不便请谅解。


