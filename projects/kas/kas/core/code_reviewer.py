"""
KAS CodeReviewer - 代码审查 Agent
自动检查代码质量、发现 bug、生成修复建议
"""
import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """代码问题"""
    file_path: str
    line_number: int
    issue_type: str  # bug, style, security, performance
    severity: str    # critical, high, medium, low
    message: str
    suggestion: str
    code_snippet: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet
        }


@dataclass
class ReviewResult:
    """审查结果"""
    file_path: str
    issues: List[CodeIssue] = field(default_factory=list)
    scanned_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "scanned_at": self.scanned_at,
            "issues_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues]
        }


class CodeReviewer:
    """
    代码审查 Agent
    
    能力:
    1. 语法错误检查 (AST解析)
    2. 常见bug模式检测
    3. 代码风格检查
    4. 安全问题扫描
    5. 生成修复建议
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.issues: List[CodeIssue] = []
        
    def review_file(self, file_path: Path) -> ReviewResult:
        """审查单个文件"""
        result = ReviewResult(file_path=str(file_path))
        
        if not file_path.exists():
            return result
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            result.issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=0,
                issue_type="bug",
                severity="high",
                message=f"无法读取文件: {e}",
                suggestion="检查文件编码和权限"
            ))
            return result
        
        # Python 文件特殊处理
        if file_path.suffix == '.py':
            result.issues.extend(self._check_python_syntax(file_path, content))
            result.issues.extend(self._check_python_patterns(file_path, content))
            result.issues.extend(self._check_imports(file_path, content))
        
        # 通用检查
        result.issues.extend(self._check_common_issues(file_path, content))
        result.issues.extend(self._check_security(file_path, content))
        
        return result
    
    def review_project(self, file_patterns: List[str] = None) -> List[ReviewResult]:
        """审查整个项目"""
        if file_patterns is None:
            file_patterns = ['*.py']
        
        results = []
        
        for pattern in file_patterns:
            for file_path in self.project_path.rglob(pattern):
                # 跳过某些目录
                if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules', 'venv', '.kas']):
                    continue
                
                result = self.review_file(file_path)
                if result.issues:
                    results.append(result)
        
        return results
    
    def _check_python_syntax(self, file_path: Path, content: str) -> List[CodeIssue]:
        """检查 Python 语法"""
        issues = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=e.lineno or 0,
                issue_type="bug",
                severity="critical",
                message=f"语法错误: {e.msg}",
                suggestion="修复语法错误",
                code_snippet=e.text or ""
            ))
        
        return issues
    
    def _check_python_patterns(self, file_path: Path, content: str) -> List[CodeIssue]:
        """检查 Python 常见 bug 模式"""
        issues = []
        lines = content.split('\n')
        
        patterns = [
            # (正则, 问题类型, 严重程度, 消息, 建议)
            (r'except\s*:', "bug", "high", "裸 except 会捕获所有异常包括 KeyboardInterrupt", "使用 except Exception:"),
            (r'==\s*(True|False|None)', "style", "low", "使用 'is' 而不是 '==' 比较单例", "将 == 改为 is"),
            (r'print\s*\(', "style", "low", "使用 logging 而不是 print", "替换为 logger.info()"),
            (r'\.format\s*\(', "style", "low", "考虑使用 f-string 提高可读性", "使用 f-string"),
            (r'open\s*\([^)]+\)(?!\s*with)', "bug", "medium", "文件未使用 with 语句，可能导致资源泄漏", "使用 with open(...) as f:"),
            (r'except.*:\s*\n\s*pass', "bug", "high", "空 except 会静默忽略错误", "至少记录错误日志"),
            (r'import\s+\*', "style", "medium", "import * 会污染命名空间", "显式导入需要的名称"),
            (r'if\s+\w+\s*==\s*\[\]', "bug", "medium", "使用 == 比较列表，应该用 if not list", "改为 if not list:"),
            (r'if\s+\w+\s*==\s*""', "bug", "medium", "使用 == 比较空字符串，应该用 if not str", "改为 if not str:"),
            (r'time\.sleep\s*\(\s*0\.001\s*\)', "performance", "low", "过短的 sleep 可能浪费 CPU", "考虑使用事件通知机制"),
            (r'while\s+True:\s*\n\s*if', "performance", "low", "考虑使用更明确的循环条件", "重构循环逻辑"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue_type, severity, message, suggestion in patterns:
                if re.search(pattern, line):
                    # 检查是否已在注释中标记为忽略
                    if '# noqa' in line or '# pylint:' in line:
                        continue
                    
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type=issue_type,
                        severity=severity,
                        message=message,
                        suggestion=suggestion,
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    def _check_imports(self, file_path: Path, content: str) -> List[CodeIssue]:
        """检查导入问题"""
        issues = []
        
        # 检查重复导入
        import_lines = {}
        for i, line in enumerate(content.split('\n'), 1):
            match = re.match(r'^(from\s+\S+\s+import|import)\s+(.+)', line.strip())
            if match:
                import_stmt = match.group(0)
                if import_stmt in import_lines:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="style",
                        severity="low",
                        message=f"重复导入: {import_stmt}",
                        suggestion="删除重复导入",
                        code_snippet=line.strip()
                    ))
                else:
                    import_lines[import_stmt] = i
        
        # 检查未使用的导入 (简单检查)
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            match = re.match(r'^from\s+(\S+)\s+import\s+(.+)', line.strip())
            if match:
                module = match.group(1)
                names = [n.strip() for n in match.group(2).split(',')]
                
                for name in names:
                    # 检查是否在文件其他地方使用 (简单统计)
                    usage_count = content.count(name) - 1  # 减去导入本身
                    if usage_count <= 0 and name != '*':
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=i,
                            issue_type="style",
                            severity="low",
                            message=f"可能未使用的导入: {name}",
                            suggestion=f"删除未使用的导入: {name}",
                            code_snippet=line.strip()
                        ))
        
        return issues
    
    def _check_common_issues(self, file_path: Path, content: str) -> List[CodeIssue]:
        """检查通用问题"""
        issues = []
        lines = content.split('\n')
        
        # 检查行长度
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="style",
                    severity="low",
                    message=f"行过长 ({len(line)} > 120 字符)",
                    suggestion="换行或重构代码",
                    code_snippet=line[:80] + "..."
                ))
        
        # 检查尾随空格
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="style",
                    severity="low",
                    message="行尾有空白字符",
                    suggestion="删除尾随空格",
                    code_snippet=repr(line[-10:])
                ))
        
        # 检查 TODO/FIXME 注释
        for i, line in enumerate(lines, 1):
            if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
                match = re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE)
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="style",
                    severity="low",
                    message=f"发现 {match.group(1)} 标记",
                    suggestion="完成或移除标记",
                    code_snippet=line.strip()
                ))
        
        return issues
    
    def _check_security(self, file_path: Path, content: str) -> List[CodeIssue]:
        """检查安全问题"""
        issues = []
        
        # 检查硬编码密码/密钥
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "可能的硬编码密码"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "可能的硬编码密钥"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "可能的硬编码 API Key"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "可能的硬编码 Token"),
            (r'eval\s*\(', "使用 eval() 有安全风险"),
            (r'exec\s*\(', "使用 exec() 有安全风险"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "shell=True 有命令注入风险"),
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="security",
                        severity="high",
                        message=message,
                        suggestion="使用环境变量或安全存储",
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    def generate_fix(self, issue: CodeIssue) -> Optional[str]:
        """为问题生成修复建议"""
        # 根据问题类型生成修复
        if "裸 except" in issue.message:
            return issue.code_snippet.replace("except Exception:", "except Exception:")
        
        if "is True" in issue.code_snippet:
            return issue.code_snippet.replace("is True", "is True")
        
        if "is False" in issue.code_snippet:
            return issue.code_snippet.replace("is False", "is False")
        
        if "is None" in issue.code_snippet:
            return issue.code_snippet.replace("is None", "is None")
        
        if "open(" in issue.code_snippet and "with" not in issue.code_snippet:
            # 简单替换，实际情况更复杂
            match = re.search(r'(\w+)\s*=\s*open\(([^)]+)\)', issue.code_snippet)
            if match:
                var_name = match.group(1)
                open_args = match.group(2)
                return f"with open({open_args}) as {var_name}:"
        
        return None
    
    def generate_report(self, results: List[ReviewResult]) -> str:
        """生成审查报告"""
        lines = [
            "# KAS 代码审查报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"项目路径: {self.project_path}",
            "\n## 统计",
        ]
        
        total_issues = sum(len(r.issues) for r in results)
        files_with_issues = len(results)
        
        severity_count = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        type_count = {"bug": 0, "style": 0, "security": 0, "performance": 0}
        
        for result in results:
            for issue in result.issues:
                severity_count[issue.severity] += 1
                type_count[issue.issue_type] += 1
        
        lines.extend([
            f"- 总问题数: {total_issues}",
            f"- 涉及文件: {files_with_issues}",
            f"\n### 严重程度分布",
            f"- 🔴 Critical: {severity_count['critical']}",
            f"- 🟠 High: {severity_count['high']}",
            f"- 🟡 Medium: {severity_count['medium']}",
            f"- 🟢 Low: {severity_count['low']}",
            f"\n### 问题类型分布",
            f"- 🐛 Bug: {type_count['bug']}",
            f"- 🎨 Style: {type_count['style']}",
            f"- 🔒 Security: {type_count['security']}",
            f"- ⚡ Performance: {type_count['performance']}",
            "\n## 详细问题",
        ])
        
        # 按严重程度排序
        for result in results:
            if result.issues:
                lines.append(f"\n### {result.file_path}")
                
                # 按严重程度排序
                sorted_issues = sorted(result.issues,
                    key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.severity])
                
                for issue in sorted_issues:
                    severity_emoji = {
                        "critical": "🔴", "high": "🟠",
                        "medium": "🟡", "low": "🟢"
                    }.get(issue.severity, "⚪")
                    
                    lines.extend([
                        f"\n**{severity_emoji} [{issue.severity.upper()}] {issue.issue_type}**",
                        f"- 位置: 第 {issue.line_number} 行",
                        f"- 问题: {issue.message}",
                        f"- 建议: {issue.suggestion}",
                    ])
                    
                    if issue.code_snippet:
                        lines.append(f"- 代码: `{issue.code_snippet}`")
                    
                    # 尝试生成修复
                    fix = self.generate_fix(issue)
                    if fix:
                        lines.append(f"- 修复: `{fix}`")
        
        return "\n".join(lines)


def run_code_review(project_path: Path, output_dir: Path) -> Path:
    """
    运行代码审查并保存结果
    
    Returns:
        报告文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行审查
    reviewer = CodeReviewer(project_path)
    results = reviewer.review_project()
    
    # 生成报告
    report = reviewer.generate_report(results)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"code_review_{timestamp}.md"
    report_path.write_text(report, encoding='utf-8')
    
    # 保存 JSON 结果
    json_path = output_dir / f"code_review_{timestamp}.json"
    json_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    logger.info(f"Code review complete: {report_path}")
    return report_path


if __name__ == "__main__":
    # 测试运行
    import sys
    
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = Path("/root/.openclaw/workspace/kas")
    
    output_dir = Path("/root/.openclaw/workspace/kas/reviews")
    report_path = run_code_review(project_path, output_dir)
    print(f"报告已生成: {report_path}")
