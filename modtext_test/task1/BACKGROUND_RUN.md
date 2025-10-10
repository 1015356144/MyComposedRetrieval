# 后台运行 CIRR 评估

这里提供了两个脚本来帮助你在后台运行CIRR多图像理解测试。

## 📋 脚本说明

### 1. `run_background.sh` - 后台启动脚本
- **参数**: `--num_groups 50 --max_pairs_per_group 6`
- **功能**: 在后台启动评估程序，并创建日志文件
- **输出**: 自动重定向到时间戳命名的日志文件

### 2. `monitor.sh` - 监控脚本  
- **功能**: 检查后台进程状态，显示日志信息
- **特性**: 实时显示进程信息、日志大小、最新输出

## 🚀 使用方法

### 启动后台评估
```bash
cd /home/guohaiyun/yangtianyu/MyComposedRetrieval/modtext_test/task1
./run_background.sh
```

### 监控进程状态
```bash
./monitor.sh
```

### 查看实时日志
```bash
# 查看输出日志
tail -f logs/eval_background_YYYYMMDD_HHMMSS.log

# 查看错误日志  
tail -f logs/eval_error_YYYYMMDD_HHMMSS.log
```

### 停止后台进程
```bash
# 方法1: 使用PID文件
kill $(cat logs/eval_background.pid)

# 方法2: 直接使用PID (从monitor.sh获取)
kill <PID>
```

## 📁 文件结构

运行后会创建以下文件结构：
```
modtext_test/task1/
├── run_background.sh          # 后台启动脚本
├── monitor.sh                 # 监控脚本  
├── logs/                      # 日志目录
│   ├── eval_background_YYYYMMDD_HHMMSS.log  # 输出日志
│   ├── eval_error_YYYYMMDD_HHMMSS.log       # 错误日志
│   └── eval_background.pid                   # 进程ID文件
└── cirr_multiimage_test_YYYYMMDD_HHMMSS.json # 结果文件
```

## ⏱️ 预估时间

基于参数 `--num_groups 50 --max_pairs_per_group 6`：
- **预计比较次数**: ~300个图像对
- **预估运行时间**: 5-10小时 (取决于硬件性能)
- **GPU内存需求**: 16GB+ 推荐

## 🔍 监控要点

1. **内存使用**: 定期检查GPU内存是否充足
2. **日志增长**: 监控日志文件大小，避免磁盘空间不足
3. **进程状态**: 使用 `monitor.sh` 定期检查进程是否正常运行
4. **错误信息**: 关注错误日志，及时发现问题

## 🛠️ 故障排除

### 常见问题
1. **GPU内存不足**: 
   - 检查其他GPU进程: `nvidia-smi`
   - 考虑使用CPU: 修改脚本添加 `--device cpu`

2. **进程意外退出**:
   - 检查错误日志: `cat logs/eval_error_*.log`
   - 查看系统日志: `dmesg | tail`

3. **磁盘空间不足**:
   - 检查可用空间: `df -h`
   - 清理旧日志文件

### 重启评估
如果需要重新开始：
```bash
# 停止当前进程
kill $(cat logs/eval_background.pid) 2>/dev/null

# 清理PID文件
rm -f logs/eval_background.pid

# 重新启动
./run_background.sh
```

## 📊 结果文件

评估完成后，结果会保存在时间戳命名的JSON文件中：
- 文件名格式: `cirr_multiimage_test_YYYYMMDD_HHMMSS.json`
- 包含所有三种方法的比较结果
- 包含详细的元数据和处理时间统计 