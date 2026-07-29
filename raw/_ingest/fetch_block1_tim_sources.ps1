<#
  fetch_block1_tim_sources.ps1
  第 1 块（训推一致性 TIM）raw 来源批量下载

  用法（在仓库根目录下运行）：
      powershell -ExecutionPolicy Bypass -File .\raw\_ingest\fetch_block1_tim_sources.ps1
  或指定仓库根：
      powershell -ExecutionPolicy Bypass -File .\fetch_block1_tim_sources.ps1 -Root "E:\97-codes\torch_parallel\llm-knowledge"

  行为：已存在的文件跳过；每条失败单独记录，末尾汇总；不会中途退出。
  说明：清单与逐条定位见同目录 INGEST_MANIFEST_block1_tim.md
#>

param(
    [string]$Root = (Get-Location).Path
)

$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------------------------------------------------------------------------
# 条目表：目标相对路径 | 下载 URL
# ---------------------------------------------------------------------------
$items = @(

    # === A. TIM 诊断与算法侧修正 =====================================
    @{ Path = 'raw\01_theory\04_posttraining\tim\Diagnosing_Training_Inference_Mismatch-2605.14220.pdf';        Url = 'https://arxiv.org/pdf/2605.14220' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\TIS_On_Rollout_Training_Mismatch-openreview_8MHqvb4lK9.pdf';   Url = 'https://openreview.net/pdf?id=8MHqvb4lK9' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\TRM_Trust_Region_Masking_SeqMIS-2512.23075.pdf';               Url = 'https://arxiv.org/pdf/2512.23075' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\ALP_Adaptive_Layerwise_Perturbation-2603.19470.pdf';           Url = 'https://arxiv.org/pdf/2603.19470' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\MIPU_Mirage_of_Optimizing_Training_Policies-2606.29526.pdf';   Url = 'https://arxiv.org/pdf/2606.29526' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\FP16_Defeating_TIM-2510.26788.pdf';                            Url = 'https://arxiv.org/pdf/2510.26788' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\Beyond_Precision_TIM_is_Optimization-2602.01826.pdf';          Url = 'https://arxiv.org/pdf/2602.01826' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\QaRL_Rollout_Aligned_QAT_RL-2604.07853.pdf';                   Url = 'https://arxiv.org/pdf/2604.07853' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\FP8_RL_Low_Precision_Stack-2601.18150.pdf';                    Url = 'https://arxiv.org/pdf/2601.18150' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\AIS_Adaptive_Importance_Sampling_Quantized_RL-2605.13907.pdf'; Url = 'https://arxiv.org/pdf/2605.13907' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\M2PO_Prosperity_before_Collapse-2510.01161.pdf';               Url = 'https://arxiv.org/pdf/2510.01161' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\VCPO_Stable_Asynchrony_ESS-2602.17616.pdf';                    Url = 'https://arxiv.org/pdf/2602.17616' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\Stabilizing_RL_with_LLMs_Qwen-2512.01374.pdf';                 Url = 'https://arxiv.org/pdf/2512.01374' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\Jackpot_Budgeted_Rejection_Sampling-2602.06107.pdf';           Url = 'https://arxiv.org/pdf/2602.06107' }
    @{ Path = 'raw\01_theory\04_posttraining\tim\RGPO_Rejection_Gated_Policy_Optimization-2604.14895.pdf';      Url = 'https://arxiv.org/pdf/2604.14895' }

    # === B. MoE 路由漂移与 RL ========================================
    @{ Path = 'raw\01_theory\04_posttraining\moe_rl\R3_Rollout_Routing_Replay-2510.11370.pdf';                  Url = 'https://arxiv.org/pdf/2510.11370' }
    @{ Path = 'raw\01_theory\04_posttraining\moe_rl\PR2_Predictive_Routing_Replay-2606.00395.pdf';              Url = 'https://arxiv.org/pdf/2606.00395' }
    @{ Path = 'raw\01_theory\04_posttraining\moe_rl\RSPO_Router_Shift_MoE_RL-2510.23027.pdf';                   Url = 'https://arxiv.org/pdf/2510.23027' }
    @{ Path = 'raw\01_theory\04_posttraining\moe_rl\CompassMax_Router_Replay_100B_MoE-2512.07710.pdf';          Url = 'https://arxiv.org/pdf/2512.07710' }

    # === C. 崩溃诊断指标 =============================================
    @{ Path = 'raw\01_theory\04_posttraining\collapse_diagnosis\AVSPO_Advantage_Collapse_ACR-2605.21125.pdf';   Url = 'https://arxiv.org/pdf/2605.21125' }
    @{ Path = 'raw\01_theory\04_posttraining\collapse_diagnosis\OPEFO_Entropy_Collapse_Entropy_Flow-2605.11491.pdf'; Url = 'https://arxiv.org/pdf/2605.11491' }

    # === D. 系统 / kernel 侧确定性 ===================================
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\determinism\TBIK_Deterministic_Inference_Across_TP-2511.17826.pdf'; Url = 'https://arxiv.org/pdf/2511.17826' }
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\determinism\LLM42_Determinism_Verified_Speculation-2601.17768.pdf'; Url = 'https://arxiv.org/pdf/2601.17768' }
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\determinism\BitExact_Inference_Verification-2606.00279.pdf';        Url = 'https://arxiv.org/pdf/2606.00279' }

    # === E. MoE 系统侧调度（第 2 块用，顺手收）=======================
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\moe_scheduling\ReLibra_Routing_Replay_Load_Balancing-2605.08639.pdf'; Url = 'https://arxiv.org/pdf/2605.08639' }
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\moe_scheduling\ForeMoE_Routing_Foresight_Microstep_LB-2606.11867.pdf'; Url = 'https://arxiv.org/pdf/2606.11867' }

    # === F. 官方文档（markdown）=====================================
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\determinism\vllm_docs_batch_invariance.md';           Url = 'https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/batch_invariance.md' }
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\determinism\sglang_docs_deterministic_inference.md';  Url = 'https://raw.githubusercontent.com/sgl-project/sglang/main/docs/advanced_features/deterministic_inference.md' }
    @{ Path = 'raw\02_engineering\04_posttrain_frameworks\determinism\miles_sglang_tim_all_in_one_blog.md';     Url = 'https://raw.githubusercontent.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/main/rlhf/slime/mismatch/blog-en.md' }
)

# ---------------------------------------------------------------------------

$ok      = @()
$skipped = @()
$failed  = @()
$total   = $items.Count
$i       = 0

Write-Host ""
Write-Host "仓库根：$Root"
Write-Host "待处理：$total 项"
Write-Host ("-" * 72)

foreach ($item in $items) {
    $i++
    $dest = Join-Path $Root $item.Path
    $name = Split-Path $dest -Leaf
    $dir  = Split-Path $dest -Parent

    if (Test-Path $dest) {
        Write-Host ("[{0,2}/{1}] SKIP  {2}" -f $i, $total, $name) -ForegroundColor DarkGray
        $skipped += $name
        continue
    }

    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    Write-Host ("[{0,2}/{1}] GET   {2}" -f $i, $total, $name) -NoNewline
    try {
        Invoke-WebRequest -Uri $item.Url -OutFile $dest -UseBasicParsing -TimeoutSec 120 `
            -UserAgent 'Mozilla/5.0 (compatible; llm-knowledge-ingest/1.0)'
        $size = (Get-Item $dest).Length
        if ($size -lt 2048) {
            Remove-Item $dest -Force
            throw "响应过小（$size 字节），疑似错误页"
        }
        Write-Host ("  OK  {0:N0} KB" -f ($size / 1KB)) -ForegroundColor Green
        $ok += $name
    }
    catch {
        Write-Host "  FAIL" -ForegroundColor Red
        Write-Host ("        {0}" -f $_.Exception.Message) -ForegroundColor DarkRed
        $failed += [PSCustomObject]@{ Name = $name; Url = $item.Url; Error = $_.Exception.Message }
        if (Test-Path $dest) { Remove-Item $dest -Force -ErrorAction SilentlyContinue }
    }

    Start-Sleep -Milliseconds 700   # 对 arXiv 客气一点
}

Write-Host ("-" * 72)
Write-Host ("完成：成功 {0} / 跳过 {1} / 失败 {2}" -f $ok.Count, $skipped.Count, $failed.Count)

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "失败清单（手动补）：" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host ("  {0}`n    {1}`n    {2}" -f $_.Name, $_.Url, $_.Error) }
}

Write-Host ""
Write-Host "另有 7 项需浏览器手动存档（博客 / issue，脚本抓不到），见 INGEST_MANIFEST_block1_tim.md 第 2 节。"
Write-Host ""
