# GenEvolve：8B SFT 断点的对齐轨迹采集

独立采集仓库，供**已经持有 Qwen3-VL-8B SFT 完整权重**的运行者使用。不训练 SFT/RL、不下载另一个基模、不调用标注或绘图模型。

默认从官方 **GenEvolve-Data-RL/train** 选 **1,000 个不同任务，每题自然采样一次**。先跑前 **50 条完整轨迹**，通过工程验收后自动续到 **总计 1,000 条**，不是额外再采 1,000 条。50 条包含在最终库中；1,000 是轨迹数量，不是决策步数量。

本仓库不包含权重、用户轨迹、参考图片或密钥。接收者只需要本仓库、已有 SFT 导出、GPU 和自己配置的 Serper 搜索服务。

## 对齐的是哪一份实现

- 工具循环、系统提示、历史拼接、工具反馈、最后一步提示：直接调用 [GenEvolve 作者公开实现 `23c847c`](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c)。**没有复制后自行改写循环**。
- 采样、token budget、tokenizer 与模型调用：直接调用 [GenEvolve-2B-Reproduction `0c35d1a`](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/0c35d1a8cce551887b9afb608afa35ca755e7104) 的 `genevolve_repro.policy.make_agent`。它支持同架构的 8B 权重；本仓库检查 8B 配置并使用已有 SFT 模型。
- 这是与上述**公开复现代码**对齐的采集器，不声称作者未发布的完整 RL 代码已经被复现。所有依赖版本见 [sources.json](sources.json)。不依赖发布者的电脑、服务器或私有目录。

### Actor 实际看到什么

Actor 的输入全是文字：原始请求、过去的 assistant 输出、工具文字反馈。`image_search` 仍会搜索并下载真实图片，但 Actor 只收到作者实现原本返回的 `IMG_###`、标题、URL、路径等文字；**不会把图片像素塞入 Actor 的历史，也不会把 `<image>` 字符串展开为图片**。图片单独保存，供后续渲染和文件验收使用。

保留原生输出格式，不换成 `<action>`：

```text
<think>reasoning</think><tool_call>{...}</tool_call>
或
<think>reasoning</think><answer>{...}</answer>
```

本轮普通 SFT 断点采集不增加词表、不插入 `<RETHINK>`、不强制触发。如果已有 tokenizer 把 `<RETHINK>` 注册成特殊 token，则仅在解码时保留它；上游解析器仍按原生 `tool_call` / `answer` 工作。其余特殊 token 的处理与训练入口一致。

**重要例外：**作者的第 11 轮提示明确要求不写 `think`、直接回答。我们原样保留，并记录 `forced_final_step=true`。这些轮次不计入“普通轮 reasoning 保留率”，也不导出为 reasoning 路由审核候选。不能把这种有意设置的省略当成模型退化。

### 采样与预算不改

| 项目 | 本仓库采集／配对 RL | 配对仓库正式 benchmark |
|---|---|---|
| temperature / top-p | 0.7 / 0.95 | 0.6 / 0.9 |
| top-k | 0，即不使用 top-k 限制 | 沿用对应评测入口 |
| 最多模型调用 | 11 | 11 |
| 每轮最多生成 token | 4,096 | 4,096 |
| 每轨迹总生成 token | 30,000 | 30,000 |
| 初始提示上限 / 上下文容量 | 6,144 / 65,536 | 沿用对应入口 |
| 最终参考图数量 | 最多 2，原生解析和排序 | 同一规则 |

这里没有将采集和 benchmark 的温度强行改成一致。采集默认 seed=42，第 i 条轨迹种子为 `42+i`；固定任务顺序、独立轨迹种子和设备列表一并锁定，续跑不重复已完成项。

## 1. 环境准备

目标环境：Linux、Python 3.11、CUDA 12.8、PyTorch 2.8.0、Transformers 4.57.1、FlashAttention 2.8.3。需 `git`、`nvcc` 和外网访问公开依赖/Hugging Face/搜索服务及图片站点。

```bash
git clone https://github.com/leiye302/genevolve-8b-trajectory-collection.git
cd genevolve-8b-trajectory-collection
bash scripts/setup.sh
```

每个 GPU 放一份 BF16 8B Actor；不会加载优化器、Teacher、图像生成模型。建议给单副本留至少 48GB 显存以容纳长历史；这不是经过真实 8B 压测的最低显存保证。显存不足会停下保留进度，不悄悄降精度、截短历史或压低预算。多卡为独立轨迹并行，不是张量并行；例如四卡设 `DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`。

## 2. 已有 SFT 权重与搜索服务

默认权重位置：`./checkpoints/qwen3-vl-8b-sft`。也可以指定现有完整 HF 导出目录；必须有模型分片、`config.json`、`tokenizer.json`、`tokenizer_config.json` 及配套 chat template。仅 LoRA adapter 或训练分布式分片不能直接作为该目录。

```bash
export SFT_CHECKPOINT=/path/to/qwen3-vl-8b-sft
export WORK=/path/to/collection-output
export DEVICES=cuda:0
read -rsp 'Serper API key: ' SERPER_API_KEY; echo
export SERPER_API_KEY
# 默认 https://google.serper.dev；若用兼容服务，必须同时支持原生 /search 和 /images 协议。
```

仅调用 Serper 搜索及网站图片下载，搜索会消费运行者的服务额度；不会调用额外 LLM API、Judge 或生图 API。不在参数、仓库或配置文件里保存真实 key。参见 [脱敏配置示例](services.env.example)。

## 3. 一键执行：先 50，验收通过后续到 1,000

```bash
bash scripts/run_collection.sh
```

离开终端仍要继续时：

```bash
nohup bash scripts/run_collection.sh > collection-console.log 2>&1 &
```

首次只下载固定版数据的三份 JSONL，用 RL train 作为任务池，用 RL eval / benchmark test 排查相同请求；**不会下载全量参考图、SFT 数据或模型**。按类别比例、固定种子取样，保留来源 ID，不按采集结果挑题。

若希望人工先看 50 条再继续：

```bash
.venv/bin/python collect.py prepare --work "$WORK"
.venv/bin/python collect.py pilot --work "$WORK" --checkpoint "$SFT_CHECKPOINT" --devices "$DEVICES"
# 检查 pilot_audit.json、episodes 和 exports；确认后执行下面一行。
.venv/bin/python collect.py run --work "$WORK" --checkpoint "$SFT_CHECKPOINT" --devices "$DEVICES"
```

`run` 对已完成的 50 条只复核、不重采；重新读取真实文件重新验收，不仅相信一个 `passed=true` 标志。

### 50 条通过条件

阈值公开在 [configs/collection.json](configs/collection.json)。这是采集工程质量门槛，**不是 benchmark 指标或语义标注准确率**：

- 50 条齐全、身份唯一，输入必须全是文字，重编码前缀与采样时保存的 token 完全一致；没有 Actor 异常或搜索 API 重试耗尽。
- 至少 80% 轨迹实际调用图片搜索；至少 80% 图片搜索调用得到已下载结果。
- 已下载文件至少 95% 可用 Pillow 完整解码；至少 80% 轨迹产出配对训练实现认定的有效最终方案。
- 原生标签格式合规率至少 95%；非强制最后一轮的非空 reasoning 比例至少 90%。

不达标就保存报告、停在预采阶段，**不会跳过失败样本补齐 50 个成功例再宣称通过**。需要排查网络、下载、模型或格式问题；若必须改模型/提示/采样/阈值，应建新工作目录重新预采，不能混入旧配置。

全量结束会再对 1,000 条做相同统计。完整文件与失败记录都保存，后续再按统一口径进行 Good/Bad 标注，不会预先把 Bad 删掉。

## 4. 输出与后续 Route-SFT 构造

```text
WORK/
  plan.json                 固定任务顺序、来源、种子
  data_contract.json        数据版本、排重记录、文件 SHA
  contract.json             模型/代码 SHA、提示词 SHA、采样合同
  episodes/*.json           每条完整轨迹，完成后原子保存
  images/trajectory_*/      搜到并实际下载的参考图
  pilot_audit.json           50 条验收报告
  full_audit.json            全量验收报告
  logs/                     每个 worker 的运行日志和失败诊断
  failed_attempts/           显式重试前归档的基础设施失败记录
  exports/
    trajectories.jsonl       完整轨迹
    decisions.jsonl          实际 H/R/原动作、前缀及输出 token
    reasoning_review.jsonl   仅 H/R 的后续审核视图
    image_manifest.jsonl     原路径→可迁移相对路径、SHA、解码结果
    manifest.json            导出数量和 SHA
```

`messages` 是每次生成前的真实快照。审核者只能依据这里的文字历史 H 与当前 R 判断：

\[
y_i = g(H_i^{\mathrm{visible}}, R_i).
\]

不能额外给审核者看图片像素，再要求只看到标题的 Actor 发现画面错误；不能把当前草稿 action、未来工具反馈或最终成败作为 reasoning 标签依据。图片本身和全部结果保存在另一份取证材料中，不拼入 `reasoning_review.jsonl`。

这是原始采集库，不是已经验收的 SFT 库。后续 Good 保留自然原动作；Bad 的纠正目标另行生成和验收。没有生成此时不可见的“正确答案”，没有自动插入标签或纠正动作。需保留整个 WORK（特别是 images），只拿 JSONL 不等于完整图像资产备份。实际历史里的路径文字不为搬家而重写；相对路径映射用于外部资产定位，不更改 Actor 曾看到的内容。

## 5. 失败与恢复

- 与固定实现相同的工具重试和格式提示；不会修改工具返回值或补图给 Actor。
- 发现 Actor/搜索基础设施错误，保存已完成证据并停本次任务；不会无上限重试消费额度。`logs/last_error_*.json` 提供脱敏错误类型、信息和代码位置。
- 原命令可续跑。模型、数据、代码、采样和设备列表不一致时拒绝混跑。工作目录锁防止误起两份 coordinator。
- 确认基础设施已修好，可显式执行 `bash scripts/run_collection.sh --retry-errors`：旧失败记录先归档，按相同 ID/seed 重试；不会重采仅仅因为答案不好、没写完或没选中参考图的轨迹。
- 不清理旧权重、不删除轨迹、不占用未指定的 GPU。

这里采的是 prompt-reference program 和工具轨迹，**不做最终生图和评分**。后续渲染继续使用已对齐仓库的官方回退：缺生成指令或可用参考图时尝试原始请求无参考图生成，失败仍计零。这种渲染回退不能拿来冒充“图片搜索和下载成功”，也不能补写到已采历史中。

## 6. 离线自检与验证范围

不用 GPU、不用 API key，不会请求真实搜索服务：

```bash
python3.11 -m venv .venv-test
source .venv-test/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python collect.py setup
python -m unittest discover -s tests -v
python collect.py --help
```

测试使用真实固定版工具循环、采样适配和图片下载代码，HTTP/模型响应由 fixture 替代，检查：文字历史与训练入口逐 token 一致、原生动作、原始图片下载和解码、最后一轮例外、marker 保留、失败阻断、重跑身份锁定、数据划分、审核视图无未来信息。

发布时仅做 CPU/模拟服务测试；**未在接收者的 8B SFT 权重和 GPU 上采出 50 条，也没有证明其搜索额度或外网能用**。仓库中的真实 50 条预采正是这一步运行时验收，不可用离线测试代替。

## 依赖与许可

上游代码通过固定提交独立下载，保留各自许可；详见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库未重新分发模型、官方数据或互联网上的图片。采集图片的使用、保留与再分发需遵循各来源的许可；不要将含私有路径、轨迹或下载资产的 WORK 直接公开提交。
