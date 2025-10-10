import json
import pandas as pd
import os
import io
import random
from PIL import Image

# 文件路径
file1_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen25_7b/R1_49/reranked_results_qwen25_7b.json"
file2_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/gpt_test/results/subset_reranked3.json"
image_base_path = "/home/guohaiyun/yty_data/CIRR/dev" 

# 输出Excel文件路径
output_excel_path1 = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/qwen25_7b_R1_but_qwen25_32b_no_top1.xlsx"
output_excel_path2 = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/qwen25_32b_R1_but_qwen25_7b_no_top1.xlsx"

# 图片参数
IMG_WIDTH = 100
IMG_HEIGHT = 100

def set_top_columns_width_px(ws, first_col, last_col, px):
    """优先像素列宽；不支持则退化为字符宽近似。"""
    try:
        ws.set_column_pixels(first_col, last_col, px)
    except Exception:
        approx_char = max(8, int(px / 7.0) + 1)  # 1字符≈7px 粗略估算
        ws.set_column(first_col, last_col, approx_char)

def load_as_50x50_bytes(img_path, size=(IMG_WIDTH, IMG_HEIGHT)):
    """将图片拉伸到指定尺寸（不保比例），返回 BytesIO。"""
    with Image.open(img_path) as im:
        im = im.convert("RGBA")
        if hasattr(Image, "Resampling"):
            im = im.resize(size, Image.Resampling.LANCZOS)
        else:
            im = im.resize(size, Image.LANCZOS)
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        bio.seek(0)
        return bio

def check_r1_result(query_data):
    """检查query是否在R1位置（即top1是否为target_hard）"""
    if not query_data['reranked_results']:
        return False
    top1_result = query_data['reranked_results'][0]['candidate_image']
    target_hard = query_data['target_hard']
    return top1_result == target_hard

def generate_excel_for_queries(query_list, data_dict1, data_dict2, output_path, title_prefix):
    """为给定的query列表生成Excel文件"""
    if not query_list:
        print(f"没有找到符合条件的query，跳过生成 {output_path}")
        return
    
    # 构建表数据
    columns = ["行类型", "query_id", "参考信息"] + [f"top{i}" for i in range(1, 11)]
    data_rows = []
    query_meta = {}
    
    for query_id in sorted(query_list):
        item1 = data_dict1[query_id]
        item2 = data_dict2[query_id]
        
        reference_image = item1['reference_image']
        target_hard = item1['target_hard']
        modification_text = item1['modification_text']
        
        # 保存元信息，便于后面插图
        query_meta[query_id] = {
            "reference_image": reference_image,
            "target_hard": target_hard,
        }
        
        # 文件1 Top10
        top10_1 = [it['candidate_image'] for it in item1['reranked_results'][:10]]
        while len(top10_1) < 10:
            top10_1.append("")
        
        # 文件2 Top10
        top10_2 = [it['candidate_image'] for it in item2['reranked_results'][:10]]
        while len(top10_2) < 10:
            top10_2.append("")
        
        # 四行基本信息
        row1 = ["查询信息", query_id, f"原图: {reference_image}\n修改文本: {modification_text}\n目标图: {target_hard}"] + [""]*10
        row2 = ["文件1 Top10", "", "qwen25_7b"] + top10_1
        row3 = ["文件2 Top10", "", "qwen25_32b"] + top10_2
        
        data_rows.extend([row1, row2, row3])
        
        # 空行分隔
        data_rows.append([""] * len(columns))
    
    # DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    
    print(f"正在生成Excel文件: {output_path}")
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=f'{title_prefix}差异比较', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets[f'{title_prefix}差异比较']
            
            # 列宽
            worksheet.set_column('A:A', 15)
            worksheet.set_column('B:B', 10)
            worksheet.set_column('C:C', 40)
            set_top_columns_width_px(worksheet, 3, 12, IMG_WIDTH + 10)
            
            # 格式
            cell_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
            
            # 表头格式
            for col_num, col_name in enumerate(columns):
                worksheet.write(0, col_num, col_name, header_format)
            
            # 写入与插图
            row_idx = 1
            total_rows = len(data_rows)
            
            while row_idx <= total_rows:
                row = data_rows[row_idx - 1] if row_idx - 1 < total_rows else None
                row_type = row[0] if row and len(row) > 0 else ""
                
                # 空行
                if not row_type:
                    for col in range(len(columns)):
                        worksheet.write(row_idx, col, "", cell_format)
                    worksheet.set_row(row_idx, 10)
                    row_idx += 1
                    continue
                
                # 查询信息行（在 D/E 列插入 原图/目标图）
                if row_type == "查询信息":
                    # 文本
                    for col in range(len(columns)):
                        worksheet.write(row_idx, col, row[col] if col < len(row) else "", cell_format)
                    
                    qid = row[1]
                    meta = query_meta.get(qid, {})
                    
                    # 在 D 列放原图
                    ref_name = meta.get("reference_image")
                    if ref_name:
                        ref_path = os.path.join(image_base_path, f"{ref_name}.png")
                        if os.path.exists(ref_path):
                            img_bytes = load_as_50x50_bytes(ref_path)
                            options = {
                                'x_offset': 3,
                                'y_offset': 3,
                                'object_position': 3,
                                'image_data': img_bytes
                            }
                            worksheet.insert_image(row_idx, 3, ref_path, options)
                        else:
                            worksheet.write(row_idx, 3, f"{ref_name}\n(图片不存在)", cell_format)
                    
                    # 在 E 列放目标图
                    tgt_name = meta.get("target_hard")
                    if tgt_name:
                        tgt_path = os.path.join(image_base_path, f"{tgt_name}.png")
                        if os.path.exists(tgt_path):
                            img_bytes = load_as_50x50_bytes(tgt_path)
                            options = {
                                'x_offset': 3,
                                'y_offset': 3,
                                'object_position': 3,
                                'image_data': img_bytes
                            }
                            worksheet.insert_image(row_idx, 4, tgt_path, options)
                        else:
                            worksheet.write(row_idx, 4, f"{tgt_name}\n(图片不存在)", cell_format)
                    
                    # 行高给足，确保图片不被裁切
                    worksheet.set_row(row_idx, IMG_HEIGHT + 15)
                    row_idx += 1
                
                # 两类图片行（Top10 结果）
                elif row_type in ["文件1 Top10", "文件2 Top10"]:
                    # 前三列文字
                    for col in range(3):
                        worksheet.write(row_idx, col, row[col] if col < len(row) else "", cell_format)
                    
                    # 图片列
                    for col in range(3, min(13, len(row))):
                        img_name = row[col]
                        worksheet.write(row_idx, col, "", cell_format)  # 边框
                        if img_name:
                            img_path = os.path.join(image_base_path, f"{img_name}.png")
                            if os.path.exists(img_path):
                                img_bytes = load_as_50x50_bytes(img_path)
                                options = {
                                    'x_offset': 3,
                                    'y_offset': 3,
                                    'object_position': 3,
                                    'image_data': img_bytes
                                }
                                worksheet.insert_image(row_idx, col, img_path, options)
                            else:
                                worksheet.write(row_idx, col, f"{img_name}\n(图片不存在)", cell_format)
                    
                    worksheet.set_row(row_idx, IMG_HEIGHT + 15)
                    row_idx += 1
                
                # 其他行（兜底）
                else:
                    for col in range(len(columns)):
                        worksheet.write(row_idx, col, row[col] if col < len(row) else "", cell_format)
                    worksheet.set_row(row_idx, 30)
                    row_idx += 1
    except ImportError:
        print("xlsxwriter未安装，使用openpyxl引擎（不支持图片插入）")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'{title_prefix}差异比较', index=False)

def main():
    # 加载数据
    print("加载文件1数据...")
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    
    print("加载文件2数据...")
    with open(file2_path, 'r') as f:
        data2 = json.load(f)
    
    # 组织数据按 query_id
    dict1 = {item['query_id']: item for item in data1['queries']}
    dict2 = {item['query_id']: item for item in data2['queries']}
    
    # 找到共有的 query_id
    common_query_ids = set(dict1.keys()) & set(dict2.keys())
    print(f"找到 {len(common_query_ids)} 个共有的查询ID")
    
    # 找出file1有R1但file2没有top1的query
    file1_r1_file2_no_top1 = []
    # 找出file2有R1但file1没有top1的query
    file2_r1_file1_no_top1 = []
    
    for query_id in common_query_ids:
        item1 = dict1[query_id]
        item2 = dict2[query_id]
        
        # 检查file1是否R1（top1 == target_hard）
        file1_is_r1 = check_r1_result(item1)
        # 检查file2是否R1（top1 == target_hard）
        file2_is_r1 = check_r1_result(item2)
        
        # file1有R1但file2没有R1的情况
        if file1_is_r1 and not file2_is_r1:
            file1_r1_file2_no_top1.append(query_id)
        
        # file2有R1但file1没有R1的情况
        if file2_is_r1 and not file1_is_r1:
            file2_r1_file1_no_top1.append(query_id)
    
    print(f"file1有R1但file2没有top1的query数量: {len(file1_r1_file2_no_top1)}")
    print(f"file2有R1但file1没有top1的query数量: {len(file2_r1_file1_no_top1)}")
    
    # 随机选择最多50个query用于生成Excel
    sample_size = 50
    
    # 生成第一个Excel：file1有R1但file2没有top1的query（随机选择50个）
    if file1_r1_file2_no_top1:
        selected_queries1 = random.sample(file1_r1_file2_no_top1, min(sample_size, len(file1_r1_file2_no_top1)))
        print(f"从{len(file1_r1_file2_no_top1)}个符合条件的query中随机选择了{len(selected_queries1)}个")
        generate_excel_for_queries(
            selected_queries1, 
            dict1, 
            dict2, 
            output_excel_path1,
            "File1有R1但File2无Top1"
        )
        print(f"Excel文件已保存至: {output_excel_path1}")
    
    # 生成第二个Excel：file2有R1但file1没有top1的query（随机选择50个）
    if file2_r1_file1_no_top1:
        selected_queries2 = random.sample(file2_r1_file1_no_top1, min(sample_size, len(file2_r1_file1_no_top1)))
        print(f"从{len(file2_r1_file1_no_top1)}个符合条件的query中随机选择了{len(selected_queries2)}个")
        generate_excel_for_queries(
            selected_queries2, 
            dict1, 
            dict2, 
            output_excel_path2,
            "File2有R1但File1无Top1"
        )
        print(f"Excel文件已保存至: {output_excel_path2}")
    
    print("分析完成！")
    print(f"图片尺寸: {IMG_WIDTH}x{IMG_HEIGHT} 像素")

if __name__ == "__main__":
    main()
