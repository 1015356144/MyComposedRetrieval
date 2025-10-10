# CIRR Multi-Image Understanding Test Tool

This tool tests Qwen2VL-7B model's multi-image understanding capabilities on the CIRR dataset by comparing images within groups using **three different approaches** to evaluate the model's visual understanding and comparison abilities.

## New Features (Updated)

- 📊 **Triple Comparison Methods**: Each image pair is tested with three different approaches
- 🌐 **English Interaction**: All model interactions are conducted in English  
- ⏰ **Timestamped Output**: Result files automatically include timestamps
- 🔄 **Method Comparison**: Direct vs Caption-based vs COT single-call comparison
- 💾 **Complete Recording**: Saves all comparison results, image IDs and processing times
- 📝 **Structured Prompts**: All methods use structured, professional prompts
- 🎯 **Concise Output**: All edit instructions are concise (around 30 words) prioritizing natural completeness

## Comparison Methods

### Method 1: Direct Structured Comparison
- Uses structured prompt with clear rules and format requirements
- Directly asks for edit instruction: "From Image 1 to Image 2: [instruction]"
- Single model call per image pair
- Fastest processing time
- Output concise (around 30 words) but naturally complete

### Method 2: Caption-Based Structured Comparison  
- Step 1: Generate detailed caption for image 1
- Step 2: Generate detailed caption for image 2
- Step 3: Compare images based on generated captions using structured prompt
- Three model calls per image pair
- More detailed analysis but longer processing time
- Output concise (around 30 words) but naturally complete

### Method 3: COT Single-Call Structured Comparison
- **NEW**: Chain-of-Thought approach in a single model call
- Step 1: Generate concise captions for both images (≤30 words each)
- Step 2: List structured differences (7 categories)
- Step 3: Compress into final edit instruction (concise, around 30 words)
- Most comprehensive analysis with structured thinking
- Single model call but longer processing time
- Includes detailed intermediate analysis

## File Structure

- `eval_found.py` - Main testing script
- `run_eval.py` - Simplified runner script with examples
- `README.md` - This documentation
- `config_example.json` - Configuration examples
- `test_setup.py` - Environment verification script

## Requirements

Make sure the following dependencies are installed:
```bash
pip install torch transformers pillow qwen-vl-utils
```

## Quick Start

### 1. Basic Usage

```bash
cd modtext_test/task1/
python eval_found.py --num_groups 1 --max_pairs_per_group 5
```

### 2. Using Runner Script

```bash
python run_eval.py
```

### 3. Environment Check

```bash
python test_setup.py
```

## Parameter Configuration

### Main Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_groups` | int | 10 | Number of image groups to test |
| `--max_pairs_per_group` | int | 5 | Maximum image pairs per group |
| `--output_file` | str | Auto-generated | Result file path (timestamp added automatically) |
| `--model_name` | str | Local path | Qwen2VL model name or path |
| `--device` | str | auto | Device selection (auto/cuda/cpu) |

### Data Path Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cirr_data_dir` | /home/guohaiyun/yty_data/CIRR/cirr | CIRR dataset directory |
| `--cirr_image_dir` | /home/guohaiyun/yty_data/CIRR | CIRR image base directory |

## Usage Examples

### Example 1: Quick Test
```bash
python eval_found.py \
    --num_groups 3 \
    --max_pairs_per_group 2 \
    --output_file quick_test.json
# Output: quick_test_20240101_123456.json
```

### Example 2: Medium Scale Test
```bash
python eval_found.py \
    --num_groups 15 \
    --max_pairs_per_group 4 \
    --device cuda
```

### Example 3: Using Different Model
```bash
python eval_found.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --num_groups 10
```

### Example 4: Custom Data Paths
```bash
python eval_found.py \
    --cirr_data_dir /custom/path/to/cirr \
    --cirr_image_dir /custom/path/to/images \
    --num_groups 5
```

## Output Format

Test results are saved in JSON format with two main sections:

### 1. Metadata
```json
{
  "metadata": {
    "test_start_time": "2024-01-01T10:00:00",
    "test_end_time": "2024-01-01T10:30:00",
    "model_name": "/path/to/qwen2vl/model",
    "device": "cuda",
    "num_groups_tested": 10,
    "max_pairs_per_group": 5,
    "total_comparisons": 45,
    "comparison_methods": {
      "method1": "Direct image comparison",
      "method2": "Caption-based comparison",
      "method3": "COT single-call comparison"
    },
    "random_seed": 42
  }
}
```

### 2. Results
```json
{
  "results": [
    {
      "group_name": "group_123",
      "image1_id": "dev-img1.jpg",
      "image2_id": "dev-img2.jpg",
      "comparison_methods": {
        "method1_direct": {
          "modification_text": "Replace sofa with bed; change wall color from white to blue.",
          "processing_time_seconds": 2.1
        },
        "method2_caption_based": {
          "image1_caption": "This image shows a living room with...",
          "image2_caption": "This image displays a bedroom with...",
          "modification_text": "Replace living room furniture with bedroom set; change color scheme.",
          "processing_time_seconds": 8.7
        },
        "method3_cot_single_call": {
          "image1_caption": "Living room with white walls and brown sofa.",
          "image2_caption": "Bedroom with blue walls and white bed.",
          "difference_list": "- Replace: sofa → bed\n- Attributes: walls: white → blue\n...",
          "modification_text": "Replace sofa with bed; change wall color white to blue.",
          "full_response": "[Step 1: Caption_Image1]...",
          "processing_time_seconds": 5.2
        }
      },
      "total_processing_time_seconds": 16.0,
      "timestamp": "2024-01-01T10:05:30"
    }
  ]
}
```

## How It Works

1. **Data Loading**: Load image annotations and split information from CIRR validation set
2. **Group Organization**: Organize related images (reference, target, etc.) into groups by pairid
3. **Random Selection**: Randomly select specified number of image groups for testing
4. **Image Comparison**: Generate all possible image pairs within each group
5. **Triple Method Analysis**: 
   - Method 1: Direct structured comparison using Qwen2VL-7B
   - Method 2: Caption generation + structured caption-based comparison
   - Method 3: **NEW** COT single-call structured analysis
6. **Result Saving**: Save all comparison results and metadata to timestamped JSON file

## Performance Considerations

- **Memory Usage**: Qwen2VL-7B requires significant GPU memory (16GB+ recommended)
- **Processing Time**: 
  - Method 1: ~2-5 seconds per pair (fastest)
  - Method 2: ~6-15 seconds per pair (3 model calls)
  - Method 3: ~4-8 seconds per pair (1 call but complex)
- **Storage Space**: Result file size depends on test scale and output length
- **Output Quality**: All methods produce concise edit instructions (around 30 words, naturally complete)

## Recommended Configurations

### Quick Test (Environment Verification)
```bash
--num_groups 2 --max_pairs_per_group 1
```
Expected time: 2-5 minutes

### Small Test (Initial Evaluation)
```bash
--num_groups 10 --max_pairs_per_group 3
```
Expected time: 20-40 minutes

### Medium Test (Comprehensive Evaluation)
```bash
--num_groups 20 --max_pairs_per_group 5
```
Expected time: 80-150 minutes

### Large Test (Full Evaluation)
```bash
--num_groups 50 --max_pairs_per_group 8
```
Expected time: 5-10 hours

## Method Comparison Analysis

### Method 1 Advantages:
- Fastest processing (single model call)
- Direct visual comparison
- Most efficient for large-scale testing
- Structured, professional prompts

### Method 2 Advantages:
- More structured analysis
- Detailed intermediate captions
- Better for understanding model's perception process
- Potentially more consistent comparisons
- Structured prompts with clear format

### Method 3 Advantages (NEW):
- **Most comprehensive**: COT provides detailed thinking process
- **Single call efficiency**: More efficient than Method 2
- **Structured analysis**: 7-category difference analysis
- **Transparency**: Shows complete reasoning chain
- **Quality**: Combines benefits of both structured prompts and detailed analysis

## Troubleshooting

### Common Issues

1. **CUDA Memory Insufficient**
   - Use `--device cpu`
   - Reduce `--max_pairs_per_group` parameter

2. **Image Files Not Found**
   - Check `--cirr_data_dir` and `--cirr_image_dir` paths
   - Ensure CIRR dataset is properly downloaded and extracted

3. **Model Loading Failed**
   - Check network connection for HuggingFace models
   - Use local model path `--model_name /path/to/model`

4. **Processing Too Slow**
   - Reduce `--num_groups` and `--max_pairs_per_group`
   - Use GPU acceleration
   - Consider testing Method 1 only for fastest results

### Debug Tips

- Start with minimal parameters: `--num_groups 1 --max_pairs_per_group 1`
- Check output logs for error messages
- Verify CIRR dataset paths and file integrity
- Run environment check: `python test_setup.py`

## Extension Features

Possible code modifications for additional functionality:

1. **Custom Prompts**: Modify structured prompts in comparison methods
2. **Batch Processing**: Optimize model inference for batch comparison
3. **Result Analysis**: Add automated analysis and statistics for results
4. **Visualization**: Generate visual reports of comparison results
5. **Method Selection**: Add option to run only specific comparison methods
6. **Word Limit Adjustment**: Modify the 30-word limit for different use cases

## Notes

- First run downloads/loads Qwen2VL model, ensure stable network connection
- Large-scale testing recommended on systems with sufficient GPU memory
- Result files can be large, ensure adequate storage space
- All model interactions are conducted in English with structured prompts
- File names automatically include timestamps to prevent overwriting
- **NEW**: All edit instructions are concise (around 30 words) prioritizing natural completeness over strict limits
- **NEW**: Method 3 provides the most detailed analysis while maintaining efficiency
- Method 3 shows complete reasoning process for better interpretability 