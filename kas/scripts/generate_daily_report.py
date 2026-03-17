#!/usr/bin/env python3
"""
生成代码审查日报
明天早上运行查看结果
"""
import json
from pathlib import Path
from datetime import datetime

def generate_daily_report():
    """生成每日审查报告"""
    reviews_dir = Path("/root/.openclaw/workspace/kas/reviews")
    
    if not reviews_dir.exists():
        print("❌ 审查目录不存在")
        return
    
    # 读取摘要历史
    summary_path = reviews_dir / "review_summary.json"
    if not summary_path.exists():
        print("❌ 没有审查记录")
        return
    
    history = json.loads(summary_path.read_text(encoding='utf-8'))
    
    if not history:
        print("❌ 没有审查记录")
        return
    
    # 统计
    total_runs = len(history)
    latest = history[-1]
    
    # 计算趋势
    if len(history) >= 2:
        prev = history[-2]
        issue_trend = latest['total_issues'] - prev['total_issues']
        fix_trend = latest['auto_fix_success']
    else:
        issue_trend = 0
        fix_trend = 0
    
    # 生成报告
    lines = [
        "# KAS 代码审查日报",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"统计周期: 过去 {total_runs} 次审查",
        "\n## 📊 总体情况",
        f"\n### 最新状态",
        f"- 扫描文件数: {latest['total_files']}",
        f"- 问题文件数: {latest['files_with_issues']}",
        f"- 总问题数: {latest['total_issues']}",
        f"- Critical: {latest['critical']}",
        f"- High: {latest['high']}",
        f"\n### 自动修复",
        f"- 修复尝试: {latest['auto_fix_attempted']}",
        f"- 修复成功: {latest['auto_fix_success']}",
    ]
    
    if fix_trend > 0:
        lines.append(f"\n✅ 本次自动修复了 {fix_trend} 个问题")
    
    # 趋势
    lines.extend([
        "\n## 📈 趋势分析",
    ])
    
    if issue_trend < 0:
        lines.append(f"- 🟢 问题数减少 {abs(issue_trend)} 个 (持续改进中)")
    elif issue_trend > 0:
        lines.append(f"- 🟠 问题数增加 {issue_trend} 个 (需要关注)")
    else:
        lines.append(f"- ⚪ 问题数保持稳定")
    
    # 最新报告链接
    latest_report = reviews_dir / "code_review_latest.md"
    if latest_report.exists():
        lines.extend([
            f"\n## 📄 详细报告",
            f"\n最新完整报告: `{latest_report}`",
            f"\n查看命令:",
            f"```bash",
            f"cat {latest_report}",
            f"```",
        ])
    
    # 历史记录
    lines.extend([
        "\n## 🕐 最近审查记录",
        "\n| 时间 | 问题数 | 修复数 | 状态 |",
        "|------|--------|--------|------|",
    ])
    
    for record in history[-10:]:  # 最近10条
        time_str = record['timestamp'][:16].replace('T', ' ')
        status = "✅" if record['critical'] == 0 else "⚠️"
        lines.append(f"| {time_str} | {record['total_issues']} | {record['auto_fix_success']} | {status} |")
    
    # 建议
    lines.extend([
        "\n## 💡 建议",
    ])
    
    if latest['critical'] > 0:
        lines.append(f"- 🔴 发现 {latest['critical']} 个 Critical 问题，需要立即处理")
    
    if latest['high'] > 0:
        lines.append(f"- 🟠 发现 {latest['high']} 个 High 优先级问题，建议今天处理")
    
    lines.append(f"- 查看详细问题列表: `cat {latest_report}`")
    lines.append(f"- 自动修复日志: `ls -la {reviews_dir}/auto_fix_*.json`")
    
    report = "\n".join(lines)
    
    # 保存日报
    daily_path = reviews_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.md"
    daily_path.write_text(report, encoding='utf-8')
    
    # 同时输出到控制台
    print(report)
    print(f"\n✅ 日报已保存: {daily_path}")
    
    return daily_path

if __name__ == "__main__":
    generate_daily_report()
