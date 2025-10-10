#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的运行脚本
"""

from generate_excel_report import generate_excel_report

# 直接使用当前目录的JSON文件生成Excel报告
if __name__ == "__main__":
    json_file = "cirr_multiimage_test_20250818_003556.json"
    
    print("开始生成Excel报告...")
    try:
        output_path = generate_excel_report(json_file)
        print(f"Excel报告生成完成: {output_path}")
    except Exception as e:
        print(f"生成报告时出错: {e}")
        import traceback
        traceback.print_exc() 