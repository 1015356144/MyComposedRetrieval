#!/usr/bin/env python3
"""
环境和功能测试脚本
用于验证CIRR多图理解测试工具的环境配置和基本功能
"""

import os
import sys
import json
import importlib

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低，建议使用Python 3.8+")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包...")
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'PIL': 'Pillow',
        'qwen_vl_utils': 'Qwen VL Utils',
        'numpy': 'NumPy',
        'tqdm': 'TQDM'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name} - 已安装")
        except ImportError:
            print(f"❌ {name} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_cirr_data():
    """检查CIRR数据集"""
    print("\n检查CIRR数据集...")
    
    data_dir = "/home/guohaiyun/yty_data/CIRR/cirr"
    image_dir = "/home/guohaiyun/yty_data/CIRR"
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"❌ CIRR数据目录不存在: {data_dir}")
        return False
    else:
        print(f"✅ CIRR数据目录存在: {data_dir}")
    
    # 检查图片目录
    if not os.path.exists(image_dir):
        print(f"❌ CIRR图片目录不存在: {image_dir}")
        return False
    else:
        print(f"✅ CIRR图片目录存在: {image_dir}")
    
    # 检查关键文件
    key_files = [
        "captions/cap.rc2.val.json",
        "image_splits/split.rc2.val.json"
    ]
    
    for file_path in key_files:
        full_path = os.path.join(data_dir, file_path)
        if not os.path.exists(full_path):
            print(f"❌ 关键文件不存在: {full_path}")
            return False
        else:
            print(f"✅ 关键文件存在: {file_path}")
    
    return True

def check_gpu_availability():
    """检查GPU可用性"""
    print("\n检查GPU可用性...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ GPU可用")
            print(f"   GPU数量: {gpu_count}")
            print(f"   GPU型号: {gpu_name}")
            print(f"   GPU内存: {gpu_memory:.1f} GB")
            
            if gpu_memory < 12:
                print("⚠️  GPU内存可能不足，建议16GB+")
            
            return True
        else:
            print("❌ GPU不可用，将使用CPU模式")
            return False
    except ImportError:
        print("❌ 无法检查GPU状态 (PyTorch未安装)")
        return False

def check_file_structure():
    """检查文件结构"""
    print("\n检查文件结构...")
    
    required_files = [
        "eval_found.py",
        "run_eval.py", 
        "README.md",
        "config_example.json"
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name} - 存在")
        else:
            print(f"❌ {file_name} - 不存在")
            return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        # 测试导入
        sys.path.append('.')
        from eval_found import CIRRMultiImageTester, parse_args
        print("✅ 主模块导入成功")
        
        # 测试参数解析
        test_args = [
            '--num_groups', '1',
            '--max_pairs_per_group', '1',
            '--output_file', 'test_output.json'
        ]
        
        # 临时修改sys.argv
        original_argv = sys.argv
        sys.argv = ['eval_found.py'] + test_args
        
        try:
            args = parse_args()
            print("✅ 参数解析成功")
            print(f"   测试组数: {args.num_groups}")
            print(f"   每组最大对数: {args.max_pairs_per_group}")
        finally:
            sys.argv = original_argv
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def generate_test_report():
    """生成测试报告"""
    print("\n" + "="*60)
    print("测试报告")
    print("="*60)
    
    checks = [
        ("Python版本", check_python_version()),
        ("依赖包", check_dependencies()),
        ("CIRR数据集", check_cirr_data()),
        ("GPU可用性", check_gpu_availability()),
        ("文件结构", check_file_structure()),
        ("基本功能", test_basic_functionality())
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\n通过检查: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有检查通过！可以开始使用测试工具。")
        print("\n建议的下一步:")
        print("1. 运行快速测试: python eval_found.py --num_groups 2 --max_pairs_per_group 1")
        print("2. 查看帮助: python run_eval.py --help")
        print("3. 查看配置示例: cat config_example.json")
    else:
        print("❌ 存在问题需要解决。请检查上面的错误信息。")
        
        failed_checks = [name for name, result in checks if not result]
        print(f"\n失败的检查: {', '.join(failed_checks)}")
    
    return passed == total

def main():
    """主函数"""
    print("CIRR多图理解测试工具 - 环境检查")
    print("="*60)
    
    success = generate_test_report()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 