import json
import pandas as pd
import os
import io
from PIL import Image

# 文件路径
file1_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/gpt_test/results/subset_reranked3.json"
file2_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen25_7b/R1_49/reranked_results_qwen25_7b.json"
file3_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen25_32b/R1_49/reranked_results1_qwen25_32b.json"
image_base_path = "/home/guohaiyun/yty_data/CIRR/dev"
output_excel_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/rerank_comparison_3files_GPT.xlsx"

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
    """将图片拉伸到 50×50（不保比例），返回 BytesIO。"""
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

# 加载数据
print("加载文件1数据...")
with open(file1_path, 'r') as f:
    data1 = json.load(f)

print("加载文件2数据...")
with open(file2_path, 'r') as f:
    data2 = json.load(f)
    
print("加载文件3数据...")
with open(file3_path, 'r') as f:
    data3 = json.load(f)

# 组织数据按 query_id
dict1 = {item['query_id']: item for item in data1['queries']}
dict2 = {item['query_id']: item for item in data2['queries']}
dict3 = {item['query_id']: item for item in data3['queries']}

# 使用file1中的所有query_id，通过id匹配其他文件
selected_query_ids = list(dict1.keys())
selected_query_ids.sort()
print(f"使用file1中的所有 {len(selected_query_ids)} 个查询ID进行分析")

# 构建表数据
columns = ["行类型", "query_id", "参考信息"] + [f"top{i}" for i in range(1, 11)]
data_rows = []
# NEW: 记录每个 query 的原图与目标图，供“查询信息”行插图用
query_meta = {}

for query_id in selected_query_ids:
    item1 = dict1[query_id]
    # 通过id匹配其他文件，如果不存在则跳过该query
    if query_id not in dict2 or query_id not in dict3:
        print(f"跳过query_id {query_id}，因为其他文件中不存在")
        continue
    
    item2 = dict2[query_id]
    item3 = dict3[query_id]

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

    # 文件3 Top10
    top10_3 = [it['candidate_image'] for it in item3['reranked_results'][:10]]
    while len(top10_3) < 10:
        top10_3.append("")

    # 四行基本信息
    # “查询信息”文本仍放在 C 列；图像会在后续写入阶段插入到 D/E 列
    row1 = ["查询信息", query_id, f"原图: {reference_image}\n修改文本: {modification_text}\n目标图: {target_hard}"] + [""]*10
    row2 = ["文件1 Top10", "", "qwen25_7b"] + top10_1
    row3 = ["文件2 Top10", "", "qwen25_32b"] + top10_2
    row4 = ["文件3 Top10", "", "single_R1_49"] + top10_3

    data_rows.extend([row1, row2, row3, row4])

    # 空行分隔
    data_rows.append([""] * len(columns))

# DataFrame
df = pd.DataFrame(data_rows, columns=columns)

print(f"正在生成Excel文件: {output_excel_path}")
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='重排结果比较', index=False)

    workbook = writer.book
    worksheet = writer.sheets['重排结果比较']

    # 列宽
    worksheet.set_column('A:A', 15)
    worksheet.set_column('B:B', 10)
    # C 列文本较多，保持较宽
    worksheet.set_column('C:C', 40)
    # D..M (3..12) 为图片列；像素宽 >= 50 + padding
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

            # 行高给足，确保 50px 图片不被裁切
            worksheet.set_row(row_idx, IMG_HEIGHT + 15)
            row_idx += 1

        # 三类图片行（Top10 结果）
        elif row_type in ["文件1 Top10", "文件2 Top10", "文件3 Top10"]:
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
                            'object_position': 3,   # 不随单元格移动或缩放
                            'image_data': img_bytes
                        }
                        worksheet.insert_image(row_idx, col, img_path, options)
                    else:
                        worksheet.write(row_idx, col, f"{img_name}\n(图片不存在)", cell_format)

            worksheet.set_row(row_idx, IMG_HEIGHT + 15)
            row_idx += 1

        # 子问题行 - 完全不展示
        elif row_type.startswith("子问题"):
            # 跳过子问题行，不写入任何内容
            continue

        # 其他行（兜底）
        else:
            for col in range(len(columns)):
                worksheet.write(row_idx, col, row[col] if col < len(row) else "", cell_format)
            worksheet.set_row(row_idx, 30)
            row_idx += 1

print(f"Excel文件已保存至: {output_excel_path}")
print(f"已在Excel中插入固定尺寸图片，图片尺寸: {IMG_WIDTH}x{IMG_HEIGHT} 像素（不随单元格拉伸变化）")
