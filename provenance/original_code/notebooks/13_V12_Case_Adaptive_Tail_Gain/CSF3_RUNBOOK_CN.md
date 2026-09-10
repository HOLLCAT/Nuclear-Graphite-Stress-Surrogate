# V12 CSF3 运行说明

## 1. 本地上传新增文件

在 Mac 本地终端执行，不要在 CSF3 终端执行：

```bash
cd "/Users/novwin/Documents/University/University of Manchester/毕设项目/Nuclear graphite/NotebookCT3"

rsync -avhPR \
  src/v12_case_adaptive_tail_gain.py \
  scripts/build_v12_case_adaptive_tail_gain_notebook.py \
  scripts/validate_v12_case_adaptive_tail_gain.py \
  cluster/run_v12_case_adaptive_tail_gain.slurm \
  notebooks/13_V12_Case_Adaptive_Tail_Gain/ \
  outputs/13_v12_case_adaptive_tail_gain/iteration_1/.gitkeep \
  README.md \
  j96317yn@csf3.itservices.manchester.ac.uk:/scratch/j96317yn/NotebookCT3/
```

`-R` 用于保留 `src/`、`scripts/`、`cluster/` 等相对目录。该命令不上传
FEM case 文件，也不会覆盖 V11/V12 以外的输出。

## 2. 远程快速验证

连接 CSF3 后执行：

```bash
cd /scratch/j96317yn/NotebookCT3
source .venv/bin/activate
python scripts/validate_v12_case_adaptive_tail_gain.py
```

应看到 `Fast V12 package validation passed`。验证只检查结构、清单、
V11 依赖、候选配置和相似组隔离，不运行 149-case 完整计算。

## 3. 提交正式任务

```bash
cd /scratch/j96317yn/NotebookCT3
source .venv/bin/activate

JOBID=$(sbatch --parsable cluster/run_v12_case_adaptive_tail_gain.slurm)
echo "$JOBID"
```

记录打印出的任务号。任务申请 8 CPU、32 GB 内存、12 小时上限。

## 4. 查看排队与运行状态

```bash
squeue -j "$JOBID" -o "%.18i %.10P %.22j %.2t %.10M %.10l %R"
```

`PD` 表示排队，`R` 表示运行。任务开始前日志文件可能还不存在。
任务开始后查看标准输出：

```bash
tail -f "slurm-ct3-v12-gain-${JOBID}.out"
```

另开一个终端查看错误日志：

```bash
tail -f "slurm-ct3-v12-gain-${JOBID}.err"
```

按 `Ctrl+C` 只会退出日志查看，不会取消集群任务。

## 5. 任务离开队列后的检查

```bash
sacct -j "$JOBID" \
  --format=JobID,JobName,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

成功状态应为 `COMPLETED` 和 `ExitCode 0:0`。然后查看：

```bash
ROOT="outputs/13_v12_case_adaptive_tail_gain/iteration_1"

cat "$ROOT/v12_complete.json"
cat "$ROOT/v12_promotion_decision.json"
column -s, -t < "$ROOT/v12_promotion_gates.csv"
column -s, -t < "$ROOT/v12_comparison_scope_metrics.csv"
cat "$ROOT/selected_deployment_formula.txt"
```

若 `promotion_status` 为 `promote_v12_adaptive_gain`，自适应增益通过全部
门槛；若为 `retain_v11_global_gain_0_90`，运行仍然成功，但正式选择会安全
回退到 V11。

## 6. 下载到本地

回到 Mac 本地终端执行：

```bash
rsync -avhP \
  j96317yn@csf3.itservices.manchester.ac.uk:/scratch/j96317yn/NotebookCT3/outputs/13_v12_case_adaptive_tail_gain/ \
  "/Users/novwin/Documents/University/University of Manchester/毕设项目/Nuclear graphite/NotebookCT3/outputs/13_v12_case_adaptive_tail_gain/"
```

下载后重点保留 `v12_complete.json`、`v12_promotion_decision.json`、
`v12_promotion_gates.csv`、`v12_comparison_scope_metrics.csv`、嵌套 OOF
增益、最终公式、两张诊断图和 executed notebook。
