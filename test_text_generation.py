"""
测试脚本：生成新的修改文本
使用composed_retrieval_dataset中的函数来测试从两张图片和原始修改文本生成新修改文本的流程
"""

import os
import sys
import torch
from PIL import Image
import argparse

# 添加src路径
sys.path.append('/home/guohaiyun/yangtianyu/MyComposedRetrieval/src')

# 导入必要的模块
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class MockModelArgs:
    """模拟模型参数"""
    def __init__(self):
        self.foundation_model_backbone = 'qwen2_vl'


class TextGenerationTester:
    """文本生成测试器"""
    
    def __init__(self, model_path="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct"):
        """初始化测试器"""
        self.model_path = model_path
        self.model_args = MockModelArgs()
        self.foundation_model = None
        self.processor = None
        
        # 加载模型
        self._load_model()
        
        # 从composed_retrieval_dataset导入生成函数
        self._import_generation_functions()
    
    def _load_model(self):
        """加载Qwen2-VL模型"""
        print(f"正在加载模型: {self.model_path}")
        
        try:
            self.foundation_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            print("✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _import_generation_functions(self):
        """导入生成函数"""
        try:
            # 尝试不同的导入路径
            try:
                from src.data.dataset.composed_retrieval_dataset import IterativeCIRRDataset
            except ImportError:
                # 添加当前目录到路径并重试
                import sys
                if '/home/guohaiyun/yangtianyu/MyComposedRetrieval' not in sys.path:
                    sys.path.insert(0, '/home/guohaiyun/yangtianyu/MyComposedRetrieval')
                from src.data.dataset.composed_retrieval_dataset import IterativeCIRRDataset
            
            # 创建一个临时的数据集实例来访问生成方法
            # 为了避免复杂的初始化，我们传入必要的最小参数
            class MockArgs:
                def __init__(self):
                    self.foundation_model_backbone = 'qwen2_vl'
                    self.model_backbone = 'qwen2_vl'
            
            class MockTrainingArgs:
                def __init__(self):
                    self.output_dir = '/tmp/test_experiment'
            
            mock_model_args = MockArgs()
            mock_training_args = MockTrainingArgs()
            
            # 创建数据集实例，但绕过复杂的数据加载
            self.dataset = IterativeCIRRDataset(
                model_args=mock_model_args,
                data_args=None,
                training_args=mock_training_args,
                iteration_round=0,
                foundation_model=self.foundation_model
            )
            
            print("✅ 生成函数导入成功")
            
        except Exception as e:
            print(f"❌ 生成函数导入失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_image(self, image_path):
        """加载图片"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            print(f"✅ 加载图片: {image_path}")
            return image
            
        except Exception as e:
            print(f"❌ 加载图片失败: {e}")
            raise
    
    def generate_new_modification_text(self, ref_image_path, target_image_path, original_text, is_hard_negative=True):
        """
        生成新的修改文本
        
        Args:
            ref_image_path: 参考图片路径
            target_image_path: 目标图片路径  
            original_text: 原始修改文本
            is_hard_negative: 是否为困难负样本上下文
            
        Returns:
            新生成的修改文本
        """
        print("\n" + "="*50)
        print("开始生成新的修改文本")
        print("="*50)
        
        # 加载图片
        ref_image = self.load_image(ref_image_path)
        target_image = self.load_image(target_image_path)
        
        print(f"📝 原始修改文本: {original_text}")
        print(f"🔄 困难负样本模式: {is_hard_negative}")
        
        device = next(self.foundation_model.parameters()).device
        
        try:
            # 调用数据集中的生成方法
            new_text = self.dataset._generate_modification_text(
                ref_image=ref_image,
                target_image=target_image,
                original_text=original_text,
                processor=self.processor,
                model_backbone=self.model_args.foundation_model_backbone,
                device=device,
                is_hard_negative=is_hard_negative
            )
            
            if new_text:
                print(f"✅ 生成成功!")
                print(f"📝 新修改文本: {new_text}")
            else:
                print("❌ 生成失败，返回None")
            
            return new_text
            
        except Exception as e:
            print(f"❌ 生成过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_batch_generation(self, test_cases):
        """
        批量测试生成
        
        Args:
            test_cases: 测试用例列表，每个元素为(ref_path, target_path, original_text)
        """
        print("\n" + "="*60)
        print("开始批量测试")
        print("="*60)
        
        results = []
        
        for i, (ref_path, target_path, original_text) in enumerate(test_cases):
            print(f"\n🧪 测试用例 {i+1}/{len(test_cases)}")
            
            # 测试困难负样本模式
            print("\n📋 困难负样本模式:")
            new_text_hard = self.generate_new_modification_text(
                ref_path, target_path, original_text, is_hard_negative=True
            )
            
            # 测试常规模式
            print("\n📋 常规多样性模式:")
            new_text_regular = self.generate_new_modification_text(
                ref_path, target_path, original_text, is_hard_negative=False
            )
            
            results.append({
                'case_id': i+1,
                'ref_image': ref_path,
                'target_image': target_path,
                'original_text': original_text,
                'hard_negative_text': new_text_hard,
                'regular_text': new_text_regular
            })
        
        # 打印汇总结果
        print("\n" + "="*60)
        print("批量测试结果汇总")
        print("="*60)
        
        for result in results:
            print(f"\n🧪 用例 {result['case_id']}:")
            print(f"   原始: {result['original_text']}")
            print(f"   困难负样本: {result['hard_negative_text']}")
            print(f"   常规模式: {result['regular_text']}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试修改文本生成')
    parser.add_argument('--ref_image', default='/home/guohaiyun/yty_data/CIRR/train/60/train-12802-0-img1.png',type=str, required=True, help='参考图片路径')
    parser.add_argument('--target_image', default='/home/guohaiyun/yty_data/CIRR/train/47/train-10105-0-img0.png',type=str, required=True, help='目标图片路径')
    parser.add_argument('--original_text', default='Pomeranian is sitting on a white surface instead of a gray one.',type=str, required=True, help='原始修改文本')
    parser.add_argument('--model_path', type=str, 
                       default="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct",
                       help='模型路径')
    parser.add_argument('--hard_negative', action='store_true', help='使用困难负样本模式')
    
    args = parser.parse_args()
    
    print("🚀 启动修改文本生成测试")
    print(f"📁 模型路径: {args.model_path}")
    
    # 创建测试器
    tester = TextGenerationTester(model_path=args.model_path)
    
    # 生成新文本
    new_text = tester.generate_new_modification_text(
        ref_image_path=args.ref_image,
        target_image_path=args.target_image,
        original_text=args.original_text,
        is_hard_negative=args.hard_negative
    )
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)


def demo_test():
    """演示测试"""
    print("🎯 运行演示测试")
    
    # 创建测试器（使用默认模型路径）
    tester = TextGenerationTester()
    
    # 定义测试用例（需要替换为实际的图片路径）
    test_cases = [
        # 示例：请替换为实际的图片路径
        ("/home/guohaiyun/yty_data/CIRR/train/60/train-12802-0-img1.png", "/home/guohaiyun/yty_data/CIRR/train/47/train-10105-0-img0.png", "Pomeranian is sitting on a white surface instead of a gray one."),
        # ("/path/to/ref2.jpg", "/path/to/target2.jpg", "add a hat"),
        # ("/path/to/ref3.jpg", "/path/to/target3.jpg", "remove the background"),
    ]
    
    if not test_cases:
        print("⚠️  请在demo_test()函数中添加实际的测试用例")
        print("示例格式:")
        print('test_cases = [')
        print('    ("/path/to/ref.jpg", "/path/to/target.jpg", "original modification text"),')
        print(']')
        return
    
    # 运行批量测试
    results = tester.test_batch_generation(test_cases)
    
    return results


if __name__ == "__main__":
    # 如果有命令行参数，运行主函数
    if len(sys.argv) > 1:
        main()
    else:
        # 否则运行演示测试
        demo_test()
