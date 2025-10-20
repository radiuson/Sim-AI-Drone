# AI Drone - FPV Drone Control with Diffusion Policies

端到端的FPV无人机AI控制系统，使用扩散策略模型从Liftoff模拟器中学习飞行控制。

> **注意**: 此项目目前仍在开发中，功能可能不稳定，API可能会发生变化。

## 特性

- **统一模型接口**: 支持多种模型架构（ResNet-UNet, TOGE）
- **LeRobot数据格式**: 标准化数据集格式，易于共享和复现
- **虚拟遥控器**: 通过Linux uinput直接控制Liftoff模拟器
- **完整训练流程**: 数据录制 → 训练 → 部署
- **实时控制**: 30+ FPS的低延迟策略推理

## 目录结构

```
ai-drone/
├── models/                    # 模型定义
├── record/                    # 数据录制
├── train/                     # 训练模块
├── deploy/                    # 部署模块
├── tools/                     # 工具
├── configs/                   # 配置文件
├── outputs/                   # 训练输出（自动生成）
├── legacy/                    # 旧代码备份
├── requirements.txt
└── README.md
```

## 开发状态

此项目目前处于积极开发阶段，以下功能正在完善中：
- 模型训练和优化
- 数据采集和处理
- 控制策略部署

## 许可证

MIT License

## 贡献

此项目目前不接受外部贡献，因为我们正在内部完善核心功能。