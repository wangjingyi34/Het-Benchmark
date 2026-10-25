# 邻接配对的独立确认试验

2026年10月25日冻结后续试验设计。原始`het.fullmodel.run.v2`、04c84a52主脚本和原calibration1—3全部保留。原始NPU块出现持续的4/6ms时段切换，因此新增邻接AB/BA确认，不删除旧run、不改用中位数掩盖原结果，也不把后续试验追称为原始预注册。

## 固定范围和执行

- 模型：ImageNet1K V1预训练ResNet18、ResNet50；输入B1/B4、3×224×224，共4个context。每context新建5个独立进程，run101—105。不可把20个进程合并为同一进程内重复。
- 使用原资产`evaluation_b1.npy`或`evaluation_b4.npy`同字节输入，原100个evaluation标注图像、同预处理和1000-way top1，记录原manifest、tensor、weights及04c helper哈希。
- 四个动作仍为stem、早期主分支、过渡主分支和过渡shortcut的eval ConvBN folding。测量全部动作数≤2的11个mask：0、1、2、3、4、5、6、8、9、10、12。
- 每mask各有独立的baseline A和candidate B模型副本；同一进程共22个模型。两侧每轮各调用1次，防止共享baseline被重复使用11倍产生缓存频率偏差。mask0为两个独立未fold基线副本，作为空操作噪声对照。
- 所有模型先各真实评价100张标注图，保存样本ID、标签、逐图预测、correct、accuracy、logit SHA256；A/B按该mask自身baseline计算retention。数值等价沿用atol=rtol=1e-4，retention阈值0.99。
- 全部quality完成后，按固定随机顺序对22个模型各warmup30次；全部warmup完成后才开始任何正式计时。计时期间不插入batch16的quality调用。
- 20个测量轮次，每轮11个mask顺序预先洗牌；每mask的20次侧顺序固定为10个AB、10个BA预先洗牌。所有顺序由device/model/batch/run组成的字符串哈希生成，计时前保存，不根据结果调整。
- 每一对的A/B相邻执行，每侧计时前后真正device synchronize。保存每侧perf_counter_ns开始/结束、UTC开始/结束、墙钟毫秒、输出shape和字节SHA256；输出复制、哈希和等价检查在两侧计时窗口都结束后执行。
- 保存每对baseline_ms、candidate_ms、原始差值与原始百分比收益，不裁剪负值，不删除慢时段；每轮结束持久化一次。失败、OOM、质量拒绝及数值不等价全部保留。
- NPU必须使用现有`npu-python`包装器，显式关闭并读回conv/matmul HF32。试验脚本检查这两个值，未关闭则拒绝。`npu-smi info`在计时前后取证，命令失败只记录，不中断试验。

## 分析边界

本轮仅用于heldout确认。原calibration1—3选择的全部方法与动作计划必须保持冻结；不能从新增配对结果重选COPA/PPO、mask6或比较策略，也不能只汇报最好的mask。完整11个mask留作统一比较和审计。

独立性单位为fresh-process block，每context n=5。每进程20个时间对只是块内重复，不是20个独立模型；100张quality图像仍是原100张，不因22个模型或5个进程重复评价而扩成新样本。原主矩阵与新确认应分别报告，再讨论一致性。

正常完整计划为20个进程、220个mask记录、4400个配对、8800个计时调用；另有22×100×20=44000条逐图质量预测。这些为计划数量，实际完成数量以原始文件为准。

记录的均值和标准差是描述性统计，不自动证明效果显著。最终比较应以每进程配对均值为单位，报告所有5个块、顺序效应、mask0噪声分布及不确定性；是否接受候选同时受真实质量与等价门约束。输出hash保留数据身份，不能从hash反推出原logit数值或替代执行凭据。

## 冻结源码及调用

测量方案冻结源码SHA256=`2f33287e7d6afffb9303724ac61186c5712cbaaa0784795196d7b1f4ec6c22bb`。首个run101已完成全部计时/质量并持久化为completed，随后末行把flush错误传给json.dumps导致终端打印失败；该原始证据保留，不重复运行。后续执行版SHA256=`e88bce1b159f018e5dfa5793c726d362b70bcf0ed9795bf2a881d0ac5c1408c5`，唯一差异是将flush移到print；计时、质量、模型和随机化未改变。单行差异与哈希重构见paired-print-fix-verification.json。

独立方法审查已核对配对时序和预算，随后补充最终可行性门（初始quality/等价、20个成功pair、所有计时输出等价同时通过）；原始失败和负收益仍全部保留。每块同时保存配对百分比均值及ratio-of-means，两者分别报告，不互相替换。

```sh
./npu-python interventions/paired_confirmation.py --device npu --model resnet18 --batch 1 --run 101 --assets assets/imagenet-val200 --output paired_raw/npu-resnet18-b1-r101.json
```

依次执行两模型、两batch、run101—105；每次由`npu-python`启动新Python进程。根协调智能体负责真实设备排队和付费实例，编写智能体不在Mac启动长时间计算。
