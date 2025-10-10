#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel报告生成器
根据JSON测试结果文件生成包含图片和修改文本的Excel报告
使用xlsxwriter库
"""

import json
import os
from pathlib import Path
import xlsxwriter
from PIL import Image
import tempfile


def resize_image(image_path, target_width=200, target_height=150):
    """
    调整图片大小并保存到临时文件
    
    Args:
        image_path: 原始图片路径
        target_width: 目标宽度
        target_height: 目标高度
    
    Returns:
        临时文件路径
    """
    try:
        with Image.open(image_path) as img:
            # 调整图片大小，保持比例
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img_resized.save(temp_file.name, 'PNG')
            return temp_file.name
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None


def generate_excel_report(json_file_path, output_excel_path=None, image_base_dir="/home/guohaiyun/yty_data/CIRR/images/dev"):
    """
    根据JSON文件生成Excel报告
    
    Args:
        json_file_path: JSON文件路径
        output_excel_path: 输出Excel文件路径，如果为None则自动生成
        image_base_dir: 图片基础目录
    """
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果没有指定输出路径，则自动生成
    if output_excel_path is None:
        json_name = Path(json_file_path).stem
        output_excel_path = f"{json_name}_report.xlsx"
    
    # 创建工作簿
    workbook = xlsxwriter.Workbook(output_excel_path)
    worksheet = workbook.add_worksheet('图片比较报告')
    
    # 定义格式
    header_format = workbook.add_format({
        'bold': True,
        'font_color': 'white',
        'bg_color': '#366092',
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True,
        'border': 1
    })
    
    text_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'border': 1
    })
    
    center_format = workbook.add_format({
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })
    
    # 设置列宽
    worksheet.set_column('A:A', 25)  # 组名
    worksheet.set_column('B:B', 30)  # Image1
    worksheet.set_column('C:C', 30)  # Image2
    worksheet.set_column('D:D', 40)  # Method1修改文本
    worksheet.set_column('E:E', 40)  # Method2修改文本
    worksheet.set_column('F:F', 40)  # Method3修改文本
    worksheet.set_column('G:G', 15)  # 处理时间
    
    # 设置表头
    headers = ['组名', 'Image 1', 'Image 2', 'Method 1\n(直接比较)', 'Method 2\n(基于描述)', 'Method 3\n(COT单次调用)', '总处理时间(秒)']
    
    # 写入表头
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    
    # 设置表头行高
    worksheet.set_row(0, 30)
    
    # 处理每个结果
    temp_files = []  # 保存临时文件路径，最后清理
    current_row = 1
    
    for result in data['results']:
        # 设置行高以适应图片
        worksheet.set_row(current_row, 120)
        
        # 组名
        worksheet.write(current_row, 0, result['group_name'], center_format)
        
        # 处理图片
        image1_id = result['image1_id']
        image2_id = result['image2_id']
        
        # 构建图片路径
        image1_path = os.path.join(image_base_dir, f"{image1_id}.png")
        image2_path = os.path.join(image_base_dir, f"{image2_id}.png")
        
        # 添加Image1
        if os.path.exists(image1_path):
            temp_img1 = resize_image(image1_path)
            if temp_img1:
                temp_files.append(temp_img1)
                # 插入图片，设置位置和大小
                worksheet.insert_image(current_row, 1, temp_img1, {
                    'x_scale': 1.0,
                    'y_scale': 1.0,
                    'x_offset': 5,
                    'y_offset': 5
                })
        else:
            worksheet.write(current_row, 1, f"图片未找到:\n{image1_path}", text_format)
        
        # 添加Image2
        if os.path.exists(image2_path):
            temp_img2 = resize_image(image2_path)
            if temp_img2:
                temp_files.append(temp_img2)
                # 插入图片，设置位置和大小
                worksheet.insert_image(current_row, 2, temp_img2, {
                    'x_scale': 1.0,
                    'y_scale': 1.0,
                    'x_offset': 5,
                    'y_offset': 5
                })
        else:
            worksheet.write(current_row, 2, f"图片未找到:\n{image2_path}", text_format)
        
        # 添加修改文本
        methods = result['comparison_methods']
        
        # Method 1 - 直接比较
        if 'method1_direct' in methods:
            method1_text = methods['method1_direct']['modification_text']
            worksheet.write(current_row, 3, method1_text, text_format)
        
        # Method 2 - 基于描述
        if 'method2_caption_based' in methods:
            method2_text = methods['method2_caption_based']['modification_text']
            worksheet.write(current_row, 4, method2_text, text_format)
        
        # Method 3 - COT单次调用
        if 'method3_cot_single_call' in methods:
            method3_text = methods['method3_cot_single_call']['modification_text']
            worksheet.write(current_row, 5, method3_text, text_format)
        
        # 总处理时间
        total_time = result.get('total_processing_time_seconds', 'N/A')
        worksheet.write(current_row, 6, total_time, center_format)
        
        current_row += 1
    
    # 添加元数据工作表
    meta_worksheet = workbook.add_worksheet('元数据')
    metadata = data['metadata']
    
    # 元数据格式
    meta_key_format = workbook.add_format({
        'bold': True,
        'bg_color': '#f0f0f0',
        'border': 1
    })
    
    meta_value_format = workbook.add_format({
        'text_wrap': True,
        'border': 1
    })
    
    meta_data = [
        ['测试开始时间', metadata.get('test_start_time', '')],
        ['测试结束时间', metadata.get('test_end_time', '')],
        ['模型名称', metadata.get('model_name', '')],
        ['设备', metadata.get('device', '')],
        ['测试组数量', metadata.get('num_groups_tested', '')],
        ['每组最大对数', metadata.get('max_pairs_per_group', '')],
        ['随机种子', metadata.get('random_seed', '')],
        ['总比较次数', metadata.get('total_comparisons', '')],
        ['', ''],
        ['比较方法说明', ''],
        ['Method 1', metadata.get('comparison_methods', {}).get('method1', '')],
        ['Method 2', metadata.get('comparison_methods', {}).get('method2', '')],
        ['Method 3', metadata.get('comparison_methods', {}).get('method3', '')],
    ]
    
    for row_idx, (key, value) in enumerate(meta_data):
        meta_worksheet.write(row_idx, 0, key, meta_key_format)
        meta_worksheet.write(row_idx, 1, value, meta_value_format)
    
    # 设置元数据工作表的列宽
    meta_worksheet.set_column('A:A', 20)
    meta_worksheet.set_column('B:B', 80)
    
    # 关闭工作簿
    workbook.close()
    print(f"Excel报告已生成: {output_excel_path}")
    
    # 清理临时文件
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return output_excel_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="根据JSON文件生成Excel报告")
    parser.add_argument("-json_file",type=str,default="cirr_multiimage_test_20250818_005431.json", help="输入的JSON文件路径")
    parser.add_argument("-o", "--output", help="输出的Excel文件路径", default=None)
    parser.add_argument("-i", "--image-dir", 
                       default="/home/guohaiyun/yty_data/CIRR/dev",
                       help="图片基础目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"错误: JSON文件不存在: {args.json_file}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"警告: 图片目录不存在: {args.image_dir}")
    
    try:
        output_path = generate_excel_report(
            json_file_path=args.json_file,
            output_excel_path=args.output,
            image_base_dir=args.image_dir
        )
        print(f"成功生成Excel报告: {output_path}")
    except Exception as e:
        print(f"生成报告时出错: {e}")


if __name__ == "__main__":
    main() 