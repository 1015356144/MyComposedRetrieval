# Excel报告生成器

这个工具可以根据JSON测试结果文件自动生成包含图片和修改文本的Excel报告。

## 安装依赖

```bash
pip install xlsxwriter Pillow
```

或者使用requirements文件：

```bash
pip install -r requirements_excel.txt
```

## 使用方法

### 方法1：直接运行脚本

```bash
python run_excel_generator.py
```

这将使用当前目录中的`cirr_multiimage_test_20250817_210440.json`文件生成Excel报告。

### 方法2：命令行参数

```bash
python generate_excel_report.py cirr_multiimage_test_20250817_210440.json
```

### 方法3：自定义参数

```bash
python generate_excel_report.py \
  cirr_multiimage_test_20250817_210440.json \
  -o custom_report.xlsx \
  -i /home/guohaiyun/yty_data/CIRR/images/dev
```

## 参数说明

- `json_file`: 输入的JSON文件路径（必需）
- `-o, --output`: 输出的Excel文件路径（可选，默认自动生成）
- `-i, --image-dir`: 图片基础目录（可选，默认为`/home/guohaiyun/yty_data/CIRR/images/dev`）

## 功能特性

1. **自动图片处理**：自动调整图片大小至统一尺寸（200x150像素）
2. **多工作表**：主报告工作表和元数据工作表
3. **格式化**：美观的表格格式，包含边框、颜色和文本自动换行
4. **图片嵌入**：将对应的图片直接嵌入到Excel单元格中
5. **文本显示**：显示三种方法的修改文本描述
6. **处理时间统计**：显示每个比较的处理时间

## 输出文件结构

生成的Excel文件包含两个工作表：

### 1. 图片比较报告（主工作表）

| 列名 | 内容 |
|------|------|
| 组名 | 测试组的名称 |
| Image 1 | 第一张图片 |
| Image 2 | 第二张图片 |
| Method 1 (直接比较) | 直接比较方法的修改文本 |
| Method 2 (基于描述) | 基于描述比较方法的修改文本 |
| Method 3 (COT单次调用) | COT单次调用方法的修改文本 |
| 总处理时间(秒) | 该比较的总处理时间 |

### 2. 元数据工作表

包含测试的元数据信息，如：
- 测试开始和结束时间
- 模型名称
- 设备信息
- 测试参数
- 比较方法说明

## 注意事项

1. 确保图片路径正确，图片文件名格式为 `{image_id}.png`
2. 程序会自动创建临时文件来调整图片大小，运行完成后会自动清理
3. 如果图片文件不存在，会在对应单元格中显示错误信息
4. 生成的Excel文件将在当前目录下创建

## 故障排除

如果遇到导入错误，请确保安装了所需的依赖包：

```bash
pip install --upgrade xlsxwriter Pillow
```

如果图片无法显示，请检查：
1. 图片路径是否正确
2. 图片文件是否存在
3. 图片文件格式是否为PNG

## 文件说明

- `generate_excel_report.py`: 主要的Excel生成器脚本
- `run_excel_generator.py`: 简单的运行脚本，使用默认参数
- `requirements_excel.txt`: 依赖包列表
- `cirr_multiimage_test_20250817_210440.json`: 测试数据文件 