#!/usr/bin/env python3
"""
KAS 自动代码审查脚本
每15分钟运行一次，检查代码问题并尝试自动修复
"""
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from kas.core.code_reviewer import CodeReviewer, run_code_review

def auto_fix_issues(project_path: Path, results: list) -> dict:
    """
    尝试自动修复发现的问题
    
    Returns:
        修复统计
    """
    fix_stats = {
        "attempted": 0,
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for result in results:
        file_path = Path(result.file_path)
        if not file_path.exists():
            continue
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            modified = False
            
            # 按行号倒序处理，避免行号变化
            sorted_issues = sorted(result.issues, key=lambda x: x.line_number, reverse=True)
            
            for issue in sorted_issues:
                fix_stats["attempted"] += 1
                
                line_idx = issue.line_number - 1
                if line_idx < 0 or line_idx >= len(lines):
                    continue
                
                original_line = lines[line_idx]
                fixed_line = None
                
                # 尝试自动修复
                if "裸 except" in issue.message:
                    fixed_line = original_line.replace("except Exception:", "except Exception:")
                
                elif "is True" in original_line:
                    fixed_line = original_line.replace("is True", "is True")
                
                elif "is False" in original_line:
                    fixed_line = original_line.replace("is False", "is False")
                
                elif "is None" in original_line:
                    fixed_line = original_line.replace("is None", "is None")
                
                elif "行尾有空白字符" in issue.message:
                    fixed_line = original_line.rstrip()
                
                elif "使用 'is' 而不是" in issue.message:
                    # 处理各种 is 比较
                    fixed_line = re.sub(r'==\s*(True|False|None)', r'is \1', original_line)
                
                # 应用修复
                if fixed_line and fixed_line != original_line:
                    lines[line_idx] = fixed_line
                    modified = True
                    fix_stats["success"] += 1
                    fix_stats["details"].append({
                        "file": str(file_path),
                        "line": issue.line_number,
                        "original": original_line.strip(),
                        "fixed": fixed_line.strip()
                    })
                else:
                    fix_stats["failed"] += 1
            
            # 保存修改
            if modified:
                backup_path = file_path.with_suffix(f".py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                file_path.rename(backup_path)
                
                file_path.write_text('\n'.join(lines), encoding='utf-8')
                print(f"  ✓ 自动修复: {file_path.name}")
        
        except Exception as e:
            print(f"  ✗ 修复失败 {file_path}: {e}")
            fix_stats["failed"] += 1
    
    return fix_stats

def main():
    """主函数"""
    print(f"\n{'='*60}")
    print(f"KAS 自动代码审查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 项目路径
    project_path = Path("/root/.openclaw/workspace/kas")
    output_dir = project_path / "reviews"
    output_dir.mkdir(exist_ok=True)
    
    # 运行审查
    print("🔍 正在扫描代码...")
    reviewer = CodeReviewer(project_path)
    results = reviewer.review_project()
    
    # 统计
    total_issues = sum(len(r.issues) for r in results)
    critical_issues = sum(1 for r in results for i in r.issues if i.severity == "critical")
    high_issues = sum(1 for r in results for i in r.issues if i.severity == "high")
    
    print(f"\n📊 扫描结果:")
    print(f"  - 扫描文件: {len(list(project_path.rglob('*.py')))}")
    print(f"  - 问题文件: {len(results)}")
    print(f"  - 总问题数: {total_issues}")
    print(f"  - 🔴 Critical: {critical_issues}")
    print(f"  - 🟠 High: {high_issues}")
    
    # 尝试自动修复
    if results:
        print(f"\n🔧 尝试自动修复...")
        fix_stats = auto_fix_issues(project_path, results)
        print(f"  - 尝试修复: {fix_stats['attempted']}")
        print(f"  - 修复成功: {fix_stats['success']}")
        print(f"  - 修复失败: {fix_stats['failed']}")
        
        # 保存修复记录
        if fix_stats['details']:
            fix_log = output_dir / f"auto_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            fix_log.write_text(json.dumps(fix_stats, indent=2, ensure_ascii=False), encoding='utf-8')
    else:
        fix_stats = {"attempted": 0, "success": 0, "failed": 0, "details": []}
    
    # 生成报告
    print(f"\n📝 生成报告...")
    report = reviewer.generate_report(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"code_review_{timestamp}.md"
    report_path.write_text(report, encoding='utf-8')
    
    # 同时保存为最新报告
    latest_path = output_dir / "code_review_latest.md"
    latest_path.write_text(report, encoding='utf-8')
    
    # 保存 JSON
    json_path = output_dir / f"code_review_{timestamp}.json"
    json_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    print(f"\n✅ 完成!")
    print(f"  报告: {report_path}")
    print(f"  最新: {latest_path}")
    
    # 摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(list(project_path.rglob('*.py'))),
        "files_with_issues": len(results),
        "total_issues": total_issues,
        "critical": critical_issues,
        "high": high_issues,
        "auto_fix_attempted": fix_stats['attempted'],
        "auto_fix_success": fix_stats['success'],
        "report_path": str(report_path)
    }
    
    summary_path = output_dir / "review_summary.json"
    # 读取历史
    history = []
    if summary_path.exists():
        try:
            history = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    
    history.append(summary)
    # 只保留最近 100 条
    history = history[-100:]
    summary_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    
    print(f"  摘要: {summary_path}")
    print(f"\n{'='*60}\n")
    
    return summary

if __name__ == "__main__":
    main()
