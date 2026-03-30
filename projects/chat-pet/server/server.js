import { WebSocketServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 内存数据库（后续可替换为 SQLite）
const db = {
  users: new Map(),
  pets: new Map(),
  messages: [],
  memories: new Map(), // petId -> []
  
  saveToFile() {
    const data = {
      users: Array.from(this.users.entries()),
      pets: Array.from(this.pets.entries()),
      messages: this.messages,
      memories: Array.from(this.memories.entries())
    };
    fs.writeFileSync(join(__dirname, 'data.json'), JSON.stringify(data, null, 2));
  },
  
  loadFromFile() {
    try {
      const data = JSON.parse(fs.readFileSync(join(__dirname, 'data.json'), 'utf8'));
      this.users = new Map(data.users);
      this.pets = new Map(data.pets);
      this.messages = data.messages || [];
      this.memories = new Map(data.memories || []);
    } catch (e) {
      // 文件不存在，使用空数据
    }
  }
};

db.loadFromFile();

// WebSocket 服务器
const wss = new WebSocketServer({ port: 8080 });
const clients = new Map(); // userId -> { ws, nickname }
const rooms = new Map();   // roomId -> Set of userIds

console.log('🐾 ChatPet Server running on ws://localhost:8080');

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
          
          db.users.set(userId, {
            id: userId,
            nickname,
            createdAt: new Date().toISOString()
          });
          
          // 创建默认宠物
          const petId = uuidv4();
          const petName = msg.petName || `${nickname}的宠物`;
          db.pets.set(petId, {
            id: petId,
            userId,
            name: petName,
            level: 1,
            exp: 0,
            personality: {
              active: 50,
              gentle: 50,
              humor: 50,
              sarcasm: 20
            },
            isAutoReply: false,
            createdAt: new Date().toISOString()
          });
          db.memories.set(petId, []);
          
          clients.set(userId, { ws, nickname });
          
          ws.send(JSON.stringify({
            type: 'registered',
            userId,
            petId
          }));
          
          console.log(`👤 新用户: ${nickname} (${userId.slice(0, 8)})`);
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
          
          // 发送历史消息
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
          
          // 广播给房间内所有人
          broadcast(currentRoom, {
            type: 'message',
            ...message
          });
          
          // 宠物观察学习
          await petObserveAndLearn(userId, currentRoom, msg.content);
          
          // 检查是否有挂机宠物需要自动回复
          await checkAutoReplies(currentRoom, userId, msg.content);
          
          db.saveToFile();
          break;
        }
        
        case 'toggle_auto_reply': {
          const userPet = Array.from(db.pets.values()).find(p => p.userId === userId);
          if (userPet) {
            userPet.isAutoReply = msg.enabled;
            db.pets.set(userPet.id, userPet);
            
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
          const pet = Array.from(db.pets.values()).find(p => p.userId === userId);
          if (pet) {
            ws.send(JSON.stringify({
              type: 'pet_status',
              pet
            }));
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

// 宠物观察学习
async function petObserveAndLearn(userId, roomId, content) {
  const pet = Array.from(db.pets.values()).find(p => p.userId === userId);
  if (!pet) return;
  
  // 提取关键词作为记忆
  const keywords = extractKeywords(content);
  const memories = db.memories.get(pet.id) || [];
  
  for (const keyword of keywords) {
    memories.push({
      id: uuidv4(),
      topic: keyword,
      content: content.slice(0, 200),
      importance: 1,
      createdAt: new Date().toISOString()
    });
  }
  
  // 限制记忆数量
  if (memories.length > 100) {
    memories.splice(0, memories.length - 100);
  }
  
  db.memories.set(pet.id, memories);
  
  // 增加经验值
  const expGain = 1 + Math.floor(content.length / 50);
  pet.exp += expGain;
  
  // 检查升级
  await checkLevelUp(pet);
  
  db.pets.set(pet.id, pet);
}

// 提取关键词
function extractKeywords(text) {
  const commonWords = new Set(['的', '了', '是', '我', '你', '在', '有', '和', '就', '不', '会', '要', '还', '这', '那', '个', '上', '也', '很', '好', '吗', '吧']);
  const words = text.split(/[\s，。！？.,!?]+/).filter(w => w.length >= 2);
  return words.filter(w => !commonWords.has(w)).slice(0, 3);
}

// 检查升级
async function checkLevelUp(pet) {
  const expNeeded = pet.level * 10;
  
  if (pet.exp >= expNeeded) {
    pet.level += 1;
    pet.exp -= expNeeded;
    
    // 升级时性格微调
    pet.personality.active = Math.min(100, pet.personality.active + Math.floor(Math.random() * 5));
    pet.personality.gentle = Math.min(100, pet.personality.gentle + Math.floor(Math.random() * 5));
    pet.personality.humor = Math.min(100, pet.personality.humor + Math.floor(Math.random() * 3));
    
    console.log(`🎉 ${pet.name} 升级到等级 ${pet.level}!`);
    
    // 通知主人
    const client = clients.get(pet.userId);
    if (client?.ws?.readyState === 1) {
      client.ws.send(JSON.stringify({
        type: 'level_up',
        pet: {
          id: pet.id,
          name: pet.name,
          level: pet.level,
          exp: pet.exp
        }
      }));
    }
  }
}

// 检查自动回复
async function checkAutoReplies(roomId, senderId, content) {
  const roomUsers = rooms.get(roomId);
  if (!roomUsers) return;
  
  for (const uid of roomUsers) {
    if (uid === senderId) continue;
    
    const pet = Array.from(db.pets.values()).find(p => p.userId === uid && p.isAutoReply);
    if (!pet) continue;
    
    // 生成回复
    const reply = await generatePetReply(pet, content);
    
    const messageId = uuidv4();
    const message = {
      id: messageId,
      roomId,
      senderId: uid,
      senderName: `${pet.name} 🤖`,
      senderType: 'pet',
      content: reply,
      isPet: true,
      timestamp: new Date().toISOString()
    };
    
    db.messages.push(message);
    
    // 广播
    broadcast(roomId, {
      type: 'message',
      ...message
    });
  }
}

// 生成宠物回复
async function generatePetReply(pet, triggerContent) {
  const memories = db.memories.get(pet.id) || [];
  const recentMemories = memories.slice(-5);
  
  // 根据性格参数生成不同风格的回复
  const templates = {
    gentle: [
      `（${pet.name}歪头看着你）我在听呢~`,
      `（${pet.name}轻轻蹭了蹭）${triggerContent.length > 10 ? '说得真详细呀' : '嗯嗯，我明白~'}`,
      `（${pet.name}眨眨眼）让我好好想想...`,
      `（${pet.name}露出温柔的表情）主人平时也这么说过呢~`
    ],
    active: [
      `（${pet.name}兴奋地跳了起来）哇！真的吗！`,
      `（${pet.name}摇着尾巴转圈）我也要参与！`,
      `（${pet.name}眼睛发亮）这个好有趣！`,
      `（${pet.name}蹦蹦跳跳）等级${pet.level}的我来回答！`
    ],
    humor: [
      `（${pet.name}推了推不存在的眼镜）根据本宠物的分析...`,
      `（${pet.name}清了清嗓子）作为一个有${pet.level}级经验的宠物...`,
      `（${pet.name}摆出严肃脸）这个问题嘛...（憋笑）`,
      `（${pet.name}突然正经）我研究这个问题已经${memories.length}秒了！`
    ],
    sarcasm: [
      `（${pet.name}翻了个白眼）哦，真有意思呢~`,
      `（${pet.name}叹气）主人，这你都说了${memories.filter(m => m.content.includes(triggerContent.slice(0, 10))).length}遍了`,
      `（${pet.name}面无表情）我在听，真的。`,
      `（${pet.name}冷笑）等级${pet.level}的我觉得这很无聊`
    ]
  };
  
  // 根据性格选择回复风格
  let style = 'gentle';
  const { active, gentle, humor, sarcasm } = pet.personality;
  
  const rand = Math.random() * 100;
  if (sarcasm > 60 && rand < 30) {
    style = 'sarcasm';
  } else if (humor > 60 && rand < 50) {
    style = 'humor';
  } else if (active > gentle && rand < 70) {
    style = 'active';
  }
  
  const replies = templates[style];
  return replies[Math.floor(Math.random() * replies.length)];
}