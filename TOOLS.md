# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

## CMOS书籍传世之作 —— 自动巡检系统

当收到cron提醒"检查CMOS书籍写作进度"时，**自动执行**：

### 标准操作流程（SOP）
1. **检查文件**：`find cmos-book/chapters -name "*.md" | wc -l`
2. **统计字数**：`wc -c` 计算总字节数
3. **读取状态**：查看 `BOOK_STATUS.md`
4. **对比记忆**：对比 `MEMORY.md` 记录
5. **更新记忆**：如有变化，立即更新 `MEMORY.md`
6. **生成报告**：向用户汇报进度

### 关键路径
- 章节文件：`/root/.openclaw/workspace/cmos-book/chapters/`
- 状态记录：`/root/.openclaw/workspace/cmos-book/BOOK_STATUS.md`
- 记忆文件：`/root/.openclaw/workspace/MEMORY.md`
- 巡检技能：`/root/.openclaw/workspace/cmos-book/.claw/cron-inspector-skill.md`

### 话术模板
```
❤️‍🔥 CMOS书籍自动巡检 —— {时间}
总体进度: [████░░░░░░] X% (N/M章)
今日变化: +X章 / +XXXX字 / 💤无变化
正在写: 第X章（如 detectable）
下一步: XXX
```

**原则**：不问用户，直接检查，3分钟完成，MEMORY.md同步。

---

Add whatever helps you do your job. This is your cheat sheet.
