#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from PIL import Image
import io
import xlsxwriter
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_data(json_file_path):
    """加载JSON数据"""
    logger.info(f"正在加载JSON数据: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"已加载 {len(data)} 个样例")
    return data

def resize_image(image_path, target_size=(200, 200)):
    """调整图片大小并保存为临时文件"""
    try:
        with Image.open(image_path) as img:
            # 保持宽高比
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # 创建白色背景
            background = Image.new('RGB', target_size, (255, 255, 255))
            
            # 计算居中位置
            x = (target_size[0] - img.width) // 2
            y = (target_size[1] - img.height) // 2
            
            # 粘贴图片到背景上
            background.paste(img, (x, y))
            
            # 创建临时文件路径
            temp_file = f"/tmp/temp_img_{hash(image_path)}_{target_size[0]}x{target_size[1]}.png"
            background.save(temp_file, format='PNG')
            
            return temp_file
    except Exception as e:
        logger.error(f"处理图片时出错 {image_path}: {str(e)}")
        # 创建空白图片
        blank_img = Image.new('RGB', target_size, (240, 240, 240))
        temp_file = f"/tmp/temp_blank_{hash(image_path)}.png"
        blank_img.save(temp_file, format='PNG')
        return temp_file



def create_summary_sheet(data, workbook):
    """创建摘要工作表"""
    ws_summary = workbook.add_worksheet("数据摘要")
    
    # 创建格式
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9E1F2',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    data_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    # 设置列宽
    ws_summary.set_column('A:A', 25)
    ws_summary.set_column('B:B', 15)
    
    # 统计信息
    total_samples = len(data)
    unique_reference_images = len(set(sample["reference_image"] for sample in data))
    unique_target_images = len(set(sample["target_image"] for sample in data))
    unique_hard_negative_images = len(set(sample["hard_negative_image"] for sample in data))
    
    # 排名位置统计
    rank_counts = {}
    for sample in data:
        rank = sample.get("rank_position", "未知")
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # 相似度分数统计
    similarity_scores = [sample.get("similarity_score", 0) for sample in data if sample.get("similarity_score") is not None]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    # 写入摘要信息
    summary_data = [
        ["统计项目", "数值"],
        ["总样例数", total_samples],
        ["唯一原图数量", unique_reference_images],
        ["唯一目标图数量", unique_target_images],
        ["唯一硬负样本图数量", unique_hard_negative_images],
        ["平均相似度分数", round(avg_similarity, 4)],
        ["", ""],
        ["排名位置分布", ""],
    ]
    
    for rank, count in sorted(rank_counts.items()):
        summary_data.append([f"排名 {rank}", count])
    
    # 写入数据
    for row_idx, row_data in enumerate(summary_data):
        for col_idx, value in enumerate(row_data):
            if row_idx == 0:
                ws_summary.write(row_idx, col_idx, value, header_format)
            else:
                ws_summary.write(row_idx, col_idx, value, data_format)

def main():
    """主函数"""
    logger.info("开始生成Excel文件")
    
    # 文件路径
    json_file_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/generate_modtext_test/hard_negatives_selected.json"
    base_image_path = "/home/guohaiyun/yty_data/CIRR"
    
    # 创建输出目录
    output_dir = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/generate_modtext_test/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"hard_negatives_visualization_{timestamp}.xlsx")
    
    # 加载数据
    data = load_json_data(json_file_path)
    
    # 创建工作簿
    workbook = xlsxwriter.Workbook(output_file)
    worksheet = workbook.add_worksheet("图片对比数据")
    
    # 创建格式
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'font_color': 'white',
        'bg_color': '#366092',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    text_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'top',
        'text_wrap': True
    })
    
    center_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    # 设置列宽
    worksheet.set_column('A:A', 20)  # 样例ID
    worksheet.set_column('B:B', 30)  # 原图
    worksheet.set_column('C:C', 30)  # 目标图
    worksheet.set_column('D:D', 30)  # 硬负样本图
    worksheet.set_column('E:E', 40)  # 修改文本
    worksheet.set_column('F:F', 12)  # 排名位置
    worksheet.set_column('G:G', 15)  # 相似度分数
    
    # 设置标题行高
    worksheet.set_row(0, 30)
    
    # 创建标题
    headers = ['样例ID', '原图 (Reference)', '目标图 (Target)', '硬负样本 (Hard Negative)', 
               '修改文本 (Modification Text)', '排名位置', '相似度分数']
    
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    
    # 存储临时文件路径，用于最后清理
    temp_files = []
    
    # 处理每个样例
    logger.info(f"开始处理 {len(data)} 个样例")
    
    for idx, sample in enumerate(data):
        row = idx + 1
        logger.info(f"处理样例 {idx + 1}/{len(data)}")
        
        # 设置数据行高
        worksheet.set_row(row, 150)
        
        # 样例ID
        worksheet.write(row, 0, f"样例 {idx + 1}", center_format)
        
        # 处理图片路径
        ref_img_path = os.path.join(base_image_path, sample["reference_image"].lstrip('./'))
        target_img_path = os.path.join(base_image_path, sample["target_image"].lstrip('./'))
        hard_neg_img_path = os.path.join(base_image_path, sample["hard_negative_image"].lstrip('./'))
        
        # 插入图片
        try:
            # 原图
            if os.path.exists(ref_img_path):
                temp_img = resize_image(ref_img_path)
                temp_files.append(temp_img)
                worksheet.insert_image(row, 1, temp_img, {
                    'x_scale': 1, 'y_scale': 1,
                    'x_offset': 5, 'y_offset': 5
                })
            else:
                worksheet.write(row, 1, "图片不存在", center_format)
                logger.warning(f"原图不存在: {ref_img_path}")
        except Exception as e:
            worksheet.write(row, 1, f"图片加载失败: {str(e)}", center_format)
            logger.error(f"原图加载失败: {ref_img_path}, 错误: {str(e)}")
        
        try:
            # 目标图
            if os.path.exists(target_img_path):
                temp_img = resize_image(target_img_path)
                temp_files.append(temp_img)
                worksheet.insert_image(row, 2, temp_img, {
                    'x_scale': 1, 'y_scale': 1,
                    'x_offset': 5, 'y_offset': 5
                })
            else:
                worksheet.write(row, 2, "图片不存在", center_format)
                logger.warning(f"目标图不存在: {target_img_path}")
        except Exception as e:
            worksheet.write(row, 2, f"图片加载失败: {str(e)}", center_format)
            logger.error(f"目标图加载失败: {target_img_path}, 错误: {str(e)}")
        
        try:
            # 硬负样本图
            if os.path.exists(hard_neg_img_path):
                temp_img = resize_image(hard_neg_img_path)
                temp_files.append(temp_img)
                worksheet.insert_image(row, 3, temp_img, {
                    'x_scale': 1, 'y_scale': 1,
                    'x_offset': 5, 'y_offset': 5
                })
            else:
                worksheet.write(row, 3, "图片不存在", center_format)
                logger.warning(f"硬负样本图不存在: {hard_neg_img_path}")
        except Exception as e:
            worksheet.write(row, 3, f"图片加载失败: {str(e)}", center_format)
            logger.error(f"硬负样本图加载失败: {hard_neg_img_path}, 错误: {str(e)}")
        
        # 修改文本
        modification_text = sample.get("modification_text", "")
        worksheet.write(row, 4, modification_text, text_format)
        
        # 排名位置
        rank_position = sample.get("rank_position", "")
        worksheet.write(row, 5, rank_position, center_format)
        
        # 相似度分数
        similarity_score = sample.get("similarity_score", "")
        worksheet.write(row, 6, similarity_score, center_format)
    
    # 添加摘要工作表
    create_summary_sheet(data, workbook)
    
    # 保存Excel文件
    logger.info(f"保存Excel文件: {output_file}")
    workbook.close()
    logger.info("Excel文件创建完成")
    
    # 清理临时文件
    logger.info("清理临时文件...")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.warning(f"删除临时文件失败 {temp_file}: {str(e)}")
    
    logger.info(f"Excel文件已生成: {output_file}")
    logger.info("处理完成！")

if __name__ == "__main__":
    main()
