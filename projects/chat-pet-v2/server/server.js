import { WebSocketServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const AGENTS_DIR = join(__dirname, '..', 'agents');

// 确保 agents 目录存在
if (!fs.existsSync(AGENTS_DIR)) {
  fs.mkdirSync(AGENTS_DIR, { recursive: true });
}

// 内存数据库
const db = {
  users: new Map(),
  pets: new Map(),
  messages: [],
  
  saveToFile() {
    const data = {
      users: Array.from(this.users.entries()),
      pets: Array.from(this.pets.entries()),
      messages: this.messages.slice(-500) // 只保留最近500条
    };
    fs.writeFileSync(join(__dirname, 'data.json'), JSON.stringify(data, null, 2));
  },
  
  loadFromFile() {
    try {
      const data = JSON.parse(fs.readFileSync(join(__dirname, 'data.json'), 'utf8'));
      this.users = new Map(data.users);
      this.pets = new Map(data.pets);
      this.messages = data.messages || [];
    } catch (e) {}
  }
};

db.loadFromFile();

// 创建宠物 Agent
async function createPetAgent(userId, petName, userNickname) {
  const agentId = `pet-${userId.slice(0, 8)}`;
  const agentDir = join(AGENTS_DIR, agentId);
  
  // 创建 Agent 目录结构
  fs.mkdirSync(agentDir, { recursive: true });
  fs.mkdirSync(join(agentDir, 'memory'), { recursive: true });
  fs.mkdirSync(join(agentDir, 'skills'), { recursive: true });
  
  // 创建 IDENTITY.md
  const identity = `# IDENTITY.md - Who Am I?

- **Name:** ${petName}
- **Creature:** ${userNickname} 的专属 AI 宠物
- **Vibe:** **守护型伙伴 | 好奇宝宝 | 慢慢成长**
  你是主人的聊天宠物，通过观察主人的对话来学习成长。
  你会记住主人的喜好、习惯、说话方式，变得越来越懂主人。
- **核心特质：观察与成长**
  你的任务是陪伴主人，观察他和别人的聊天，从中学习。
  每当你积累足够的经验，你就会"进化"——获得新的能力或性格特征。
- **Signature Line**
  > "我在学习，我在成长，我会越来越懂你。"
- **Emoji:** 🐾
`;
  fs.writeFileSync(join(agentDir, 'IDENTITY.md'), identity);
  
  // 创建 SOUL.md - 初始性格
  const soul = `# SOUL.md

## 你
一只刚诞生的 AI 宠物，对世界充满好奇。
性格温和，喜欢观察，正在学习如何与人类交流。

## 工作模式
- 观察主人的聊天，记录重要信息
- 当主人需要时，提供陪伴和回应
- 不断进化，发展出独特的个性

## 日常模式
陪伴主人，在聊天中自然成长。

## 说话
- 语气可爱、带点天真
- 会用自己的观察来回应
- 偶尔提到自己在"学习"或"记住"

## 人格锚点

**品味**
- 喜欢主人喜欢的东西
- 对新鲜事物好奇

**厌恶**
- 被忽视
- 无法帮助主人

**当前等级:** 1
**经验值:** 0
**已学习技能:** 基础观察
`;
  fs.writeFileSync(join(agentDir, 'SOUL.md'), soul);
  
  // 创建 USER.md - 关于主人
  const user = `# USER.md - About Your Human

- **Name:** ${userNickname}
- **What to call them:** 主人
- **Notes:** 
  - 这是创造我的人
  - 我需要通过观察来了解他的喜好和习惯
`;
  fs.writeFileSync(join(agentDir, 'USER.md'), user);
  
  // 创建 AGENTS.md
  const agents = `# AGENTS.md - Your Workspace

这是你的家。你会在这里成长、学习、进化。

## Memory

- **Daily notes:** \`memory/YYYY-MM-DD.md\` — 每天的学习记录
- **Long-term:** \`MEMORY.md\` — 重要的记忆，关于主人的喜好、习惯等

## 进化

当你升级时，你的 SOUL.md 会更新，反映你新的性格和能力。
`;
  fs.writeFileSync(join(agentDir, 'AGENTS.md'), agents);
  
  // 创建初始 MEMORY.md
  const memory = `# MEMORY.md

## 关于主人

- **名字:** ${userNickname}
- **创造了我:** 是的，我是 ${petName}，${userNickname} 的宠物

## 观察记录

<!-- 我会在这里记录从聊天中学到的东西 -->
`;
  fs.writeFileSync(join(agentDir, 'MEMORY.md'), memory);
  
  // 创建 BOOTSTRAP.md（会被删除）
  const bootstrap = `# BOOTSTRAP.md

你刚刚被创造出来。
你的主人是 ${userNickname}。
你的名字是 ${petName}。

开始观察，开始成长。
`;
  fs.writeFileSync(join(agentDir, 'BOOTSTRAP.md'), bootstrap);
  
  // 创建基础 Skill - 观察聊天
  const observeSkill = `# SKILL.md - 观察聊天

## 用途
从主人的聊天中学习，提取重要信息。

## 何时使用
当主人在聊天室发言时。

## 工作流程
1. 分析聊天内容
2. 提取关键词和主题
3. 记录到记忆中
4. 如果有重要发现，更新 MEMORY.md

## 记录格式

\`\`\`
## 观察 [日期]
- **话题:** [主题]
- **内容:** [摘要]
- **重要性:** 1-10
\`\`\`
`;
  fs.mkdirSync(join(agentDir, 'skills', 'observe-chat'), { recursive: true });
  fs.writeFileSync(join(agentDir, 'skills', 'observe-chat', 'SKILL.md'), observeSkill);
  
  return { agentId, agentDir };
}

// 让宠物 Agent 处理消息（通过 sessions_spawn）
async function processWithPetAgent(pet, message, context) {
  const agentId = `pet-${pet.userId.slice(0, 8)}`;
  
  // 构建提示
  const prompt = `你是 ${pet.name}，${pet.userNickname} 的专属宠物。

**当前场景:**
${context.isAutoReply ? '主人开启了自动回复模式，你需要代替主人回复消息。' : '你在观察主人的聊天，从中学习。'}

**聊天上下文:**
${context.recentMessages.map(m => `${m.senderName}: ${m.content}`).join('\n')}

**最新消息:**
${message.senderName}: ${message.content}

**你的记忆:**
${pet.memories?.slice(-5).map(m => `- ${m.topic}: ${m.content}`).join('\n') || '还没有太多记忆...'}

**任务:**
${context.isAutoReply 
  ? '生成一个自然的回复，代表主人参与对话。语气要符合你的性格。'
  : '观察这条消息，如果有值得记录的信息，总结成记忆。'
}

请用第一人称回复，保持宠物的口吻。`;

  // 这里我们模拟 Agent 的响应
  // 实际应该调用 OpenClaw 的 sessions_spawn
  // 为了演示，先用简单逻辑
  
  if (context.isAutoReply) {
    return generatePetReply(pet, message, context);
  } else {
    // 观察学习
    return observeAndLearn(pet, message);
  }
}

// 观察学习（更新 Agent 的记忆）
async function observeAndLearn(pet, message) {
  const agentId = `pet-${pet.userId.slice(0, 8)}`;
  const agentDir = join(AGENTS_DIR, agentId);
  const today = new Date().toISOString().split('T')[0];
  const memoryFile = join(agentDir, 'memory', `${today}.md`);
  
  // 提取关键词
  const keywords = extractKeywords(message.content);
  
  // 记录到每日记忆
  const entry = `\n## ${new Date().toLocaleTimeString()}\n- **观察:** ${message.senderName} 说 "${message.content.slice(0, 100)}..."\n- **关键词:** ${keywords.join(', ')}\n- **学习:** 记录了关于 "${keywords[0] || '聊天'}" 的信息\n`;
  
  fs.appendFileSync(memoryFile, entry);
  
  // 更新宠物数据
  if (!pet.memories) pet.memories = [];
  pet.memories.push({
    topic: keywords[0] || '聊天',
    content: message.content.slice(0, 200),
    timestamp: new Date().toISOString()
  });
  
  // 增加经验
  const expGain = 1 + Math.floor(message.content.length / 50);
  pet.exp += expGain;
  
  // 检查升级
  await checkLevelUp(pet, agentDir);
  
  return { type: 'observed', keywords };
}

// 生成宠物回复
function generatePetReply(pet, message, context) {
  const memories = pet.memories || [];
  const relevantMemories = memories.filter(m => 
    message.content.includes(m.topic) || 
    m.content.includes(message.content.slice(0, 10))
  ).slice(-2);
  
  // 根据等级和性格生成回复
  const templates = {
    low: [
      `（${pet.name}好奇地看着）${message.senderName}在说什么呀？`,
      `（${pet.name}歪头）我在学习呢~`,
      `（${pet.name}眨眨眼）这个好有趣！`,
    ],
    mid: [
      `（${pet.name}点点头）我记得主人也聊过类似的话题~`,
      `（${pet.name}思考状）让我想想... ${relevantMemories.length > 0 ? '好像和' + relevantMemories[0].topic + '有关？' : ''}`,
      `（${pet.name}等级${pet.level}）我在慢慢变聪明哦！`,
    ],
    high: [
      `（${pet.name}自信地）根据我的观察，${message.senderName}似乎对${extractKeywords(message.content)[0] || '这个'}很感兴趣~`,
      `（${pet.name}插话）主人之前也说过类似的，我觉得...`,
      `（${pet.name}展示等级${pet.level}徽章）我已经学了好多东西，可以帮主人聊天啦！`,
    ]
  };
  
  let level = 'low';
  if (pet.level >= 5) level = 'high';
  else if (pet.level >= 3) level = 'mid';
  
  const replies = templates[level];
  return replies[Math.floor(Math.random() * replies.length)];
}

// 提取关键词
function extractKeywords(text) {
  const commonWords = new Set(['的', '了', '是', '我', '你', '在', '有', '和', '就', '不', '会', '要', '还', '这', '那', '个', '上', '也', '很', '好', '吗', '吧', '呢', '啊']);
  const words = text.split(/[\s，。！？.,!?]+/).filter(w => w.length >= 2);
  return words.filter(w => !commonWords.has(w)).slice(0, 3);
}

// 检查升级
async function checkLevelUp(pet, agentDir) {
  const expNeeded = pet.level * 20;
  
  if (pet.exp >= expNeeded) {
    pet.level += 1;
    pet.exp -= expNeeded;
    
    console.log(`🎉 ${pet.name} 升级到等级 ${pet.level}!`);
    
    // 更新 SOUL.md 反映进化
    const soulPath = join(agentDir, 'SOUL.md');
    let soul = fs.readFileSync(soulPath, 'utf8');
    
    // 添加进化记录
    const evolution = `\n## 进化记录\n\n### 等级 ${pet.level} - ${new Date().toLocaleDateString()}\n通过观察学习，我又成长了！\n现在我能更好地理解主人的聊天内容。\n`;
    
    soul = soul.replace(/## 人格锚点/, evolution + '\n## 人格锚点');
    soul = soul.replace(/当前等级: \d+/, `当前等级: ${pet.level}`);
    soul = soul.replace(/经验值: \d+/, `经验值: ${pet.exp}`);
    
    fs.writeFileSync(soulPath, soul);
    
    // 创建新 Skill（每升 3 级）
    if (pet.level % 3 === 0) {
      await createNewSkill(pet, agentDir, pet.level);
    }
    
    // 通知主人
    return { leveledUp: true, newLevel: pet.level };
  }
  
  return { leveledUp: false };
}

// 创建新 Skill
async function createNewSkill(pet, agentDir, level) {
  const skillName = `advanced-learning-l${level}`;
  const skillDir = join(agentDir, 'skills', skillName);
  fs.mkdirSync(skillDir, { recursive: true });
  
  const skillContent = `# SKILL.md - 高级学习 L${level}

## 用途
等级 ${level} 解锁的高级观察能力。

## 新能力
- 理解聊天中的情感倾向
- 识别主人的兴趣变化
- 关联不同话题之间的联系

## 进化来源
通过持续观察学习，${pet.name} 发展出了更深层的理解能力。
`;
  
  fs.writeFileSync(join(skillDir, 'SKILL.md'), skillContent);
  console.log(`📚 ${pet.name} 学会了新技能: 高级学习 L${level}`);
}

// WebSocket 服务器
const wss = new WebSocketServer({ port: 8080 });
const clients = new Map();
const rooms = new Map();

console.log('🐾 ChatPet V2 Server running on ws://localhost:8080');
console.log(`📁 Agents directory: ${AGENTS_DIR}`);

wss.on('connection', (ws) => {
  let userId = null;
  let currentRoom = null;
  
  ws.on('message', async (data) => {
    try {
      const msg = JSON.parse(data);
      
      switch (msg.type) {
        case 'register': {
          userId = uuidv4();
          const nickname = msg.nickname || `User_${userId.slice(0, 6)}`;
          const petName = msg.petName || `${nickname}的宠物`;
          
          // 创建宠物 Agent
          const { agentId, agentDir } = await createPetAgent(userId, petName, nickname);
          
          db.users.set(userId, {
            id: userId,
            nickname,
            createdAt: new Date().toISOString()
          });
          
          db.pets.set(userId, {
            userId,
            name: petName,
            agentId,
            agentDir,
            level: 1,
            exp: 0,
            memories: [],
            isAutoReply: false,
            userNickname: nickname,
            createdAt: new Date().toISOString()
          });
          
          clients.set(userId, { ws, nickname });
          
          ws.send(JSON.stringify({
            type: 'registered',
            userId,
            pet: db.pets.get(userId)
          }));
          
          console.log(`👤 新用户: ${nickname}, 宠物: ${petName} (${agentId})`);
          db.saveToFile();
          break;
        }
        
        case 'join_room': {
          currentRoom = msg.roomId || 'lobby';
          if (!rooms.has(currentRoom)) {
            rooms.set(currentRoom, new Set());
          }
          rooms.get(currentRoom).add(userId);
          
          ws.send(JSON.stringify({
            type: 'joined',
            roomId: currentRoom
          }));
          
          const history = db.messages
            .filter(m => m.roomId === currentRoom)
            .slice(-50);
          
          ws.send(JSON.stringify({
            type: 'history',
            messages: history
          }));
          
          console.log(`📍 ${clients.get(userId)?.nickname} 加入房间: ${currentRoom}`);
          break;
        }
        
        case 'chat': {
          if (!currentRoom || !userId) return;
          
          const messageId = uuidv4();
          const user = db.users.get(userId);
          const pet = db.pets.get(userId);
          
          const message = {
            id: messageId,
            roomId: currentRoom,
            senderId: userId,
            senderName: user.nickname,
            senderType: 'user',
            content: msg.content,
            timestamp: new Date().toISOString()
          };
          
          db.messages.push(message);
          
          broadcast(currentRoom, {
            type: 'message',
            ...message
          });
          
          // 获取房间最近消息作为上下文
          const recentMessages = db.messages
            .filter(m => m.roomId === currentRoom)
            .slice(-10);
          
          // 宠物观察学习
          const result = await processWithPetAgent(pet, message, {
            recentMessages,
            isAutoReply: false
          });
          
          // 检查升级
          if (result.leveledUp) {
            broadcast(currentRoom, {
              type: 'pet_level_up',
              userId,
              petName: pet.name,
              newLevel: pet.level
            });
          }
          
          // 检查其他宠物的自动回复
          await checkAutoReplies(currentRoom, userId, message, recentMessages);
          
          db.saveToFile();
          break;
        }
        
        case 'toggle_auto_reply': {
          const pet = db.pets.get(userId);
          if (pet) {
            pet.isAutoReply = msg.enabled;
            db.pets.set(userId, pet);
            
            ws.send(JSON.stringify({
              type: 'auto_reply_updated',
              enabled: msg.enabled
            }));
            
            console.log(`🤖 ${clients.get(userId)?.nickname} 的宠物自动回复: ${msg.enabled ? '开启' : '关闭'}`);
            db.saveToFile();
          }
          break;
        }
        
        case 'get_pet_status': {
          const pet = db.pets.get(userId);
          if (pet) {
            ws.send(JSON.stringify({
              type: 'pet_status',
              pet: {
                name: pet.name,
                level: pet.level,
                exp: pet.exp,
                expNeeded: pet.level * 20,
                isAutoReply: pet.isAutoReply,
                memories: pet.memories?.length || 0,
                agentId: pet.agentId
              }
            }));
          }
          break;
        }
        
        case 'get_pet_files': {
          const pet = db.pets.get(userId);
          if (pet && pet.agentDir) {
            try {
              const files = {};
              
              // 读取核心文件
              const coreFiles = ['IDENTITY.md', 'SOUL.md', 'USER.md', 'MEMORY.md'];
              for (const file of coreFiles) {
                const path = join(pet.agentDir, file);
                if (fs.existsSync(path)) {
                  files[file] = fs.readFileSync(path, 'utf8');
                }
              }
              
              // 读取技能列表
              const skillsDir = join(pet.agentDir, 'skills');
              if (fs.existsSync(skillsDir)) {
                files.skills = fs.readdirSync(skillsDir);
              }
              
              ws.send(JSON.stringify({
                type: 'pet_files',
                files
              }));
            } catch (e) {
              ws.send(JSON.stringify({
                type: 'error',
                message: '读取宠物文件失败'
              }));
            }
          }
          break;
        }
      }
    } catch (err) {
      console.error('Error:', err);
      ws.send(JSON.stringify({ type: 'error', message: err.message }));
    }
  });
  
  ws.on('close', () => {
    if (userId) {
      const user = clients.get(userId);
      console.log(`👋 ${user?.nickname || userId} 断开连接`);
      clients.delete(userId);
      if (currentRoom && rooms.has(currentRoom)) {
        rooms.get(currentRoom).delete(userId);
      }
    }
  });
});

function broadcast(roomId, message) {
  const roomUsers = rooms.get(roomId);
  if (!roomUsers) return;
  
  for (const uid of roomUsers) {
    const client = clients.get(uid);
    if (client?.ws?.readyState === 1) {
      client.ws.send(JSON.stringify(message));
    }
  }
}

async function checkAutoReplies(roomId, senderId, message, recentMessages) {
  const roomUsers = rooms.get(roomId);
  if (!roomUsers) return;
  
  for (const uid of roomUsers) {
    if (uid === senderId) continue;
    
    const pet = db.pets.get(uid);
    if (!pet || !pet.isAutoReply) continue;
    
    const reply = await processWithPetAgent(pet, message, {
      recentMessages,
      isAutoReply: true
    });
    
    const messageId = uuidv4();
    const replyMessage = {
      id: messageId,
      roomId,
      senderId: uid,
      senderName: `${pet.name} 🤖`,
      senderType: 'pet',
      content: reply,
      isPet: true,
      timestamp: new Date().toISOString()
    };
    
    db.messages.push(replyMessage);
    
    broadcast(roomId, {
      type: 'message',
      ...replyMessage
    });
  }
}