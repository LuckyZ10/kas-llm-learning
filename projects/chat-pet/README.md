# ChatPet - 聊天宠物

一个带 AI 宠物的实时聊天软件。每个用户都有一个专属宠物，宠物会观察学习聊天记录，慢慢进化。

## 功能

- ✅ 多用户实时聊天（WebSocket）
- ✅ 每个用户有专属宠物
- ✅ 宠物观察学习（记录关键词记忆）
- ✅ 等级/经验值系统
- ✅ 性格参数（活泼、温柔、幽默、毒舌）
- ✅ 挂机自动回复模式
- ✅ 本地数据持久化

## 启动

```bash
# 1. 启动服务器
cd server
npm install
node server.js

# 2. 打开客户端
# 用浏览器打开 client/index.html
# 或者启动本地服务器:
cd client
npx serve .
```

## 使用

1. 打开网页，输入昵称和宠物名字
2. 进入聊天室开始聊天
3. 宠物会自动学习你的聊天内容
4. 开启"自动回复"让宠物帮你聊天
5. 宠物会随着聊天慢慢升级、性格进化

## 项目结构

```
chat-pet/
├── server/
│   ├── server.js      # WebSocket 服务器
│   ├── package.json
│   └── data.json      # 本地数据存储
└── client/
    └── index.html     # 聊天界面
```