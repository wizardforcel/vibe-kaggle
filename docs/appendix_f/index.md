# 附录F：Kaggle MCP 配置与安全实践

> MCP（Model Context Protocol）让 Agent 能持续操作 Kaggle 平台：下载数据、跑训练、存工件、提交比赛。本附录讲怎么配置，以及怎么把安全护栏焊死。

---

## F.1 为什么需要 MCP

没有 MCP，Agent 只能“聊天 + 写代码”，你得手动复制粘贴、手动跑、手动提交。有了 MCP，Agent 能真正“动手”：一条流水线从读数据干到提交。

但“能动手”是把双刃剑。所以 MCP 的核心不是“能做什么”，而是“**安全地能做什么**”。

---

## F.2 最小配置示例

```json
// mcp_config.json（示意）
{
  "mcpServers": {
    "kaggle": {
      "command": "python",
      "args": ["-m", "kaggle_mcp_server"],
      "env": {
        "KAGGLE_USERNAME": "your_username",
        "KAGGLE_KEY": "your_key"
      }
    }
  }
}
```

**凭证安全**：
- `kaggle.json` 放在 `~/.kaggle/`，权限设为仅自己可读；
- 绝不把 key 写进代码或提交到 Git；
- 用环境变量注入，不硬编码。

---

## F.3 五条安全护栏（全书核心）

| 护栏 | 实现 | 作用 |
|------|------|------|
| 预览-确认双调用 | `dry_run=True` 默认 | 提交前先看，不真交 |
| 每日提交预算 | `daily_submit_limit: 3` | 防 LB 过拟合 |
| 不可逆操作默认关 | `allow_destructive: false` | 防误删 |
| 工具白名单 | 只暴露 `train/predict/submit` | 防乱执行 |
| 审计日志 | `mcp_audit.jsonl` | 可追溯 |

```yaml
# configs/agent_policy.yaml
submit_budget:
  daily_limit: 3
  total_limit: 50
  require_human_approval: true

allowed_tools:
  - "read_data"
  - "write_artifact"
  - "train_model"
  - "predict"
  - "submit_dryrun"
  - "submit"

forbidden_tools:
  - "shell_exec"
  - "pip_install"
  - "wget"
  - "delete_file"

data_access:
  allow_external: false
  forbidden_paths:
    - "~/.kaggle"
    - "~/.ssh"
```

---

## F.4 提交工具（带 dry-run）

```python
async def kaggle_submit(req):
    if req.dry_run:
        return {"success": True, "message": f"[预演] {req.file_path}", "dry_run": True}
    # 真实提交（调 Kaggle API）
    return {"success": True, "submission_id": "12345", "dry_run": False}
```

---

## F.5 安全实践清单

- [ ] 凭证只放环境变量 / `~/.kaggle/`，不进 Git；
- [ ] 提交默认 `dry_run=True`，人确认后才 `False`；
- [ ] 每日提交预算写进 `agent_policy.yaml`；
- [ ] 禁止 `shell_exec` / `pip_install` / `wget`；
- [ ] 所有 MCP 调用记审计日志；
- [ ] 定期轮换 Kaggle API key。

下一附录是 Kaggle 评估指标速查。