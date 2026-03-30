# ChatPet V2 - AI 宠物进化系统

每个宠物就是一个独立的 OpenClaw Agent，通过观察学习不断进化。

## 核心概念

- **宠物 = Agent**: 每个宠物有自己的目录结构（IDENTITY.md, SOUL.md, USER.md, MEMORY.md）
- **进化 = 更新 Skill**: 升级时创建新的 SKILL.md，发展出独特能力
- **记忆 = 文件系统**: 观察记录写入 memory/YYYY-MM-DD.md，长期记忆写入 MEMORY.md
- **性格 = SOUL.md**: 随着等级提升，SOUL.md 会更新反映新的性格特征

## 启动

```bash
cd server
npm install
node server.js

# 打开 client/index.html
```

## 宠物文件结构

```
agents/pet-{userId}/
├── IDENTITY.md      # 宠物身份
├── SOUL.md          # 性格与进化记录
├── USER.md          # 关于主人的信息
├── MEMORY.md        # 长期记忆
├── AGENTS.md        # 工作空间说明
├── memory/          # 每日观察记录
│   ├── 2026-03-01.md
│   └── ...
└── skills/          # 学到的技能
    ├── observe-chat/
    │   └── SKILL.md
    └── advanced-learning-l3/
        └── SKILL.md
```

## 进化机制

1. **观察学习**: 主人每发一条消息，宠物提取关键词记录到记忆
2. **经验值**: 消息越长、内容越丰富，获得经验越多
3. **升级**: 每升一级，SOUL.md 更新，性格参数微调
4. **新技能**: 每升 3 级，自动创建新的 Skill 目录

## 自动回复

开启后，宠物会根据：
- 当前等级（回复风格不同）
- 已有记忆（关联相关话题）
- 聊天上下文

生成自然的回复。