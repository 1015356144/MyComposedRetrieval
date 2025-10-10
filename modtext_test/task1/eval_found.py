import os
import json
import argparse
import random
from typing import List, Dict, Tuple, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIRR Multi-Image Understanding Test")
    
    # Data related parameters
    parser.add_argument('--cirr_data_dir', type=str, 
                       default='/home/guohaiyun/yty_data/CIRR/cirr',
                       help='CIRR dataset directory path')
    parser.add_argument('--cirr_image_dir', type=str,
                       default='/home/guohaiyun/yty_data/CIRR',
                       help='CIRR image base directory path')
    
    # Model related parameters
    parser.add_argument('--model_name', type=str,
                       default='/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct',
                       help='Qwen2VL model name or path')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device selection (auto/cuda/cpu)')
    
    # Test parameters
    parser.add_argument('--num_groups', type=int, default=1,
                       help='Number of groups to test')
    parser.add_argument('--max_pairs_per_group', type=int, default=3,
                       help='Maximum number of image pairs per group')
    parser.add_argument('--output_file', type=str,
                       default=None,
                       help='Result save file path (will add timestamp if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


class CIRRMultiImageTester:
    """CIRR Multi-Image Understanding Capability Tester"""
    
    def __init__(self, args):
        self.args = args
        self.setup_random_seed()
        self.setup_device()
        self.setup_output_file()
        self.load_model_and_processor()
        self.load_cirr_data()
        
    def setup_random_seed(self):
        """Set random seed"""
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
        print(f"Random seed set: {self.args.seed}")
    
    def setup_device(self):
        """Setup device"""
        if self.args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.args.device
        print(f"Using device: {self.device}")
    
    def setup_output_file(self):
        """Setup output file with timestamp"""
        if self.args.output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"cirr_multiimage_test_{timestamp}.json"
        else:
            # Add timestamp to provided filename
            base_name, ext = os.path.splitext(self.args.output_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"{base_name}_{timestamp}{ext}"
        
        print(f"Output file: {self.output_file}")
    
    def load_model_and_processor(self):
        """Load model and processor"""
        print(f"Loading model: {self.args.model_name}")
        try:
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(
                self.args.model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                trust_remote_code=True
            )
            
            # Load model
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                trust_remote_code=True
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def load_cirr_data(self):
        """Load CIRR dataset"""
        print("Loading CIRR dataset...")
        
        # Load validation set data
        captions_file = os.path.join(self.args.cirr_data_dir, 'captions/cap.rc2.val.json')
        splits_file = os.path.join(self.args.cirr_data_dir, 'image_splits/split.rc2.val.json')
        
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"CIRR annotation file not found: {captions_file}")
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"CIRR splits file not found: {splits_file}")
        
        # Load annotation data
        with open(captions_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load image split information
        with open(splits_file, 'r') as f:
            self.image_splits = json.load(f)
        
        # Organize data by groups, create image groups
        self.image_groups = self._organize_by_groups()
        
        print(f"Loading completed:")
        print(f"  - Annotations: {len(self.annotations)} entries")
        print(f"  - Total images: {len(self.image_splits)} images")
        print(f"  - Image groups: {len(self.image_groups)} groups")
    
    def _organize_by_groups(self) -> Dict[str, List[str]]:
        """Organize image data by groups"""
        groups = {}
        
        # Extract group information from annotation data
        for ann in self.annotations:
            pairid = ann.get('pairid', str(random.randint(1000, 9999)))
            group_key = f"group_{pairid}"
            
            if group_key not in groups:
                groups[group_key] = set()
            
            # Add reference and target images
            if 'reference' in ann:
                groups[group_key].add(ann['reference'])
            if 'target_hard' in ann:
                groups[group_key].add(ann['target_hard'])
            
            # Add other related images if available
            if 'img_set' in ann and 'members' in ann['img_set']:
                for member in ann['img_set']['members']:
                    groups[group_key].add(member)
        
        # Convert to list and filter groups with less than 2 images
        filtered_groups = {}
        for group_key, images in groups.items():
            image_list = list(images)
            if len(image_list) >= 2:  # Need at least 2 images for comparison
                filtered_groups[group_key] = image_list
        
        return filtered_groups
    
    def load_image(self, image_name: str) -> Image.Image:
        """Load image"""
        if image_name in self.image_splits:
            relative_path = self.image_splits[image_name]
            if relative_path.startswith('./'):
                relative_path = relative_path[2:]
            full_path = os.path.join(self.args.cirr_image_dir, relative_path)
        else:
            # If not found in splits, try direct path concatenation
            full_path = os.path.join(self.args.cirr_image_dir, image_name)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")
        
        return Image.open(full_path).convert('RGB')
    
    def generate_single_image_caption(self, image: Image.Image) -> str:
        """Generate caption for a single image"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": "Please provide an accurate description of this image."
                        }
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def compare_images_directly(self, image1: Image.Image, image2: Image.Image) -> str:
        """Method 1: Direct comparison of two images"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image1,
                        },
                        {
                            "type": "image", 
                            "image": image2,
                        },
                        {
                            "type": "text",
                            "text": """You are an image-difference analyst. Compare these two images and provide a concise edit instruction that transforms Image 1 into Image 2.

Rules:
- Base everything ONLY on visible evidence. No speculation.
- Use precise, plain English. Use concrete counts (e.g., "1 → 2"). Prefer common color terms.
- Generate a CONCISE edit instruction (around 30 words, but natural completeness is more important than strict limits).

Output Format:
[Edit Instruction]
From Image 1 to Image 2: <1-2 command-style sentences using patterns like: "Replace A with B; move A from B to C; change color from A to B; change count from A to B; set background to X." Avoid opinions. Be concise but complete.>"""
                        }
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Reduced for concise response
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Extract edit instruction from response
            extracted_instruction = self._extract_edit_instruction(output_text)
            
            return extracted_instruction
            
        except Exception as e:
            return f"Error in direct comparison: {str(e)}"
    
    def compare_images_via_captions(self, image1: Image.Image, image2: Image.Image) -> Dict[str, str]:
        """Method 2: Compare images via generated captions"""
        try:
            # Generate captions for both images
            print("      Generating caption for image 1...")
            caption1 = self.generate_single_image_caption(image1)
            
            print("      Generating caption for image 2...")
            caption2 = self.generate_single_image_caption(image2)
            
            # Compare using captions
            print("      Comparing via captions...")
            comparison_prompt = f"""You are an image-difference analyst. Based on these image descriptions, provide a concise edit instruction that transforms Image 1 into Image 2.

Image 1 Description: {caption1}
Image 2 Description: {caption2}

Rules:
- Base everything ONLY on the provided descriptions. No speculation.
- Use precise, plain English. Use concrete counts (e.g., "1 → 2"). Prefer common color terms.
- Generate a CONCISE edit instruction (around 30 words, but natural completeness is more important than strict limits).

Output Format:
[Edit Instruction]
From Image 1 to Image 2: <1-2 command-style sentences using patterns like: "Replace A with B; move A from B to C; change color from A to B; change count from A to B; set background to X." Avoid opinions. Be concise but complete.>"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": comparison_prompt
                        }
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Reduced for concise response
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            comparison_result = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Extract edit instruction from response
            extracted_instruction = self._extract_edit_instruction(comparison_result)
            
            return {
                'caption1': caption1,
                'caption2': caption2,
                'comparison': extracted_instruction
            }
            
        except Exception as e:
            return {
                'caption1': f"Error generating caption 1: {str(e)}",
                'caption2': f"Error generating caption 2: {str(e)}",
                'comparison': f"Error in caption-based comparison: {str(e)}"
            }
    
    def compare_images_cot_single_call(self, image1: Image.Image, image2: Image.Image) -> Dict[str, str]:
        """Method 3: COT (Chain of Thought) single call comparison"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image1,
                        },
                        {
                            "type": "image", 
                            "image": image2,
                        },
                        {
                            "type": "text",
                            "text": """You are an image-difference analyst and edit-instruction expert. In a SINGLE response, do three stages: (1) write a concise caption for each image; (2) compare the two captions and list structured differences; (3) compress the differences into a clear edit instruction that transforms Image 1 into Image 2.

Rules:
- Base everything ONLY on visible evidence. No speculation; 
- Use precise, plain English. Use concrete counts (e.g., "1 → 2"). Prefer common color terms.
- Generate a CONCISE final edit instruction (around 30 words, but natural completeness is more important than strict limits).

Input:
- Image 1: the first attached image.
- Image 2: the second attached image.

Thinking & Output Format (print these section headers verbatim):

[Step 1: Caption_Image1]
Provide an accurate description of this image.

[Step 2: Caption_Image2]
Provide an accurate description of this image.

[Step 3: Difference List]
- Add/Remove: <new or removed subjects>
- Replace: <A → B class/category swaps>
- Attributes: <object>: <from> → <to> (color/material/size/orientation/pose/count)
- Position/Layout: <from> → <to> (relative placement, crop, framing)
- Text/Signage: <transcribe exact text if visible; note changes; use (uncertain) if unsure>
- Lighting/Style/Background: <from> → <to>
- Other: <important changes not covered above>

[FINAL Edit Instruction]
From Image 1 to Image 2: <1-2 command-style sentences using patterns like: "Replace A with B; move A from B to C; change color from A to B; change count from A to B; set background to X." Avoid opinions. Be concise but complete.>"""
                        }
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,  # Keep higher for structured response
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Parse the structured response
            response_parts = self._parse_structured_cot_response(output_text)
            
            return response_parts
            
        except Exception as e:
            return {
                'image1_caption': f"Error in COT comparison: {str(e)}",
                'image2_caption': f"Error in COT comparison: {str(e)}",
                'difference_list': f"Error in COT comparison: {str(e)}",
                'modification_text': f"Error in COT comparison: {str(e)}",
                'full_response': f"Error in COT comparison: {str(e)}"
            }
    
    def _extract_edit_instruction(self, response: str) -> str:
        """Extract edit instruction from response text"""
        try:
            # Look for the edit instruction section
            lines = response.split('\n')
            instruction = ""
            
            for line in lines:
                line_stripped = line.strip()
                if '[edit instruction]' in line.lower():
                    continue
                elif line_stripped.lower().startswith('from image 1 to image 2:'):
                    instruction = line_stripped
                    break
                elif 'from image 1 to image 2' in line_stripped.lower():
                    instruction = line_stripped
                    break
            
            if instruction:
                # Clean up the instruction
                if ':' in instruction:
                    instruction = instruction.split(':', 1)[1].strip()
                
                # More flexible word limit - allow up to 40 words, prefer natural endings
                words = instruction.split()
                if len(words) > 80:
                    # Only truncate if significantly over limit, look for natural endings
                    truncated = ' '.join(words[:40])
                    # Look for sentence endings in the last 10 words
                    for i in range(35, 40):
                        if i < len(words) and words[i].endswith(('.', ';', '!')):
                            truncated = ' '.join(words[:i+1])
                            break
                    instruction = truncated
                else:
                    # Keep original if 40 words or less
                    instruction = ' '.join(words)
                
                return instruction
            else:
                # If no structured instruction found, be more lenient
                words = response.strip().split()
                if len(words) > 40:
                    # Look for natural cutoff in first 40 words
                    truncated = ' '.join(words[:40])
                    for i in range(35, 40):
                        if i < len(words) and words[i].endswith(('.', ';', '!')):
                            truncated = ' '.join(words[:i+1])
                            break
                    return truncated
                return response.strip()
                
        except Exception:
            # If extraction fails, be more lenient
            words = response.strip().split()
            if len(words) > 40:
                truncated = ' '.join(words[:40])
                # Try to end at a natural point
                for i in range(35, 40):
                    if i < len(words) and words[i].endswith(('.', ';', '!')):
                        truncated = ' '.join(words[:i+1])
                        break
                return truncated
            return response.strip()
    
    def _parse_structured_cot_response(self, response: str) -> Dict[str, str]:
        """Parse structured COT response into separate components"""
        try:
            # Initialize variables
            image1_caption = ""
            image2_caption = ""
            difference_list = ""
            final_instruction = ""
            
            # Split response into lines for processing
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line_lower = line.lower().strip()
                
                # Detect section headers
                if '[step 1: caption_image1]' in line_lower:
                    current_section = 'caption1'
                    continue
                elif '[step 2: caption_image2]' in line_lower:
                    current_section = 'caption2'
                    continue
                elif '[step 3: difference list]' in line_lower:
                    current_section = 'differences'
                    continue
                elif '[final edit instruction]' in line_lower:
                    current_section = 'final'
                    continue
                
                # Collect content for each section
                if line.strip() and current_section:
                    if current_section == 'caption1':
                        image1_caption += line.strip() + " "
                    elif current_section == 'caption2':
                        image2_caption += line.strip() + " "
                    elif current_section == 'differences':
                        difference_list += line.strip() + "\n"
                    elif current_section == 'final':
                        final_instruction += line.strip() + " "
            
            # Process final instruction - more flexible word limit
            processed_instruction = final_instruction.strip()
            if processed_instruction:
                # Look for "From Image 1 to Image 2:" pattern
                if 'from image 1 to image 2:' in processed_instruction.lower():
                    parts = processed_instruction.split(':', 1)
                    if len(parts) > 1:
                        processed_instruction = parts[1].strip()
                
                # More flexible word limit - allow up to 40 words
                words = processed_instruction.split()
                if len(words) > 40:
                    # Only truncate if significantly over limit
                    truncated = ' '.join(words[:40])
                    # Look for natural sentence endings in the last 10 words
                    for i in range(35, 40):
                        if i < len(words) and words[i].endswith(('.', ';', '!')):
                            truncated = ' '.join(words[:i+1])
                            break
                    processed_instruction = truncated
                else:
                    # Keep original if 40 words or less
                    processed_instruction = ' '.join(words)
            
            return {
                'image1_caption': image1_caption.strip() if image1_caption else "Could not extract image 1 caption",
                'image2_caption': image2_caption.strip() if image2_caption else "Could not extract image 2 caption",
                'difference_list': difference_list.strip() if difference_list else "Could not extract difference list",
                'modification_text': processed_instruction if processed_instruction else "Could not extract final instruction",
                'full_response': response
            }
            
        except Exception as e:
            # If parsing fails, be more lenient with word limit
            words = response.strip().split()
            if len(words) > 40:
                truncated_response = ' '.join(words[:40])
                # Try to end naturally
                for i in range(35, 40):
                    if i < len(words) and words[i].endswith(('.', ';', '!')):
                        truncated_response = ' '.join(words[:i+1])
                        break
            else:
                truncated_response = response.strip()
            
            return {
                'image1_caption': "Parsing failed",
                'image2_caption': "Parsing failed", 
                'difference_list': "Parsing failed",
                'modification_text': truncated_response,
                'full_response': response
            }
    
    def _parse_cot_response(self, response: str) -> Dict[str, str]:
        """Parse COT response into structured parts (legacy method, replaced by _parse_structured_cot_response)"""
        # This method is kept for backward compatibility but now delegates to the new parser
        return self._parse_structured_cot_response(response)
    
    def compare_two_images(self, image1_name: str, image2_name: str) -> Dict:
        """Compare two images using all three methods"""
        try:
            # Load images
            image1 = self.load_image(image1_name)
            image2 = self.load_image(image2_name)
            
            print(f"    Method 1: Direct comparison...")
            start_time = time.time()
            direct_result = self.compare_images_directly(image1, image2)
            direct_time = time.time() - start_time
            
            print(f"    Method 2: Caption-based comparison...")
            start_time = time.time()
            caption_result = self.compare_images_via_captions(image1, image2)
            caption_time = time.time() - start_time
            
            print(f"    Method 3: COT single-call comparison...")
            start_time = time.time()
            cot_result = self.compare_images_cot_single_call(image1, image2)
            cot_time = time.time() - start_time
            
            return {
                'method1_direct': {
                    'modification_text': direct_result,
                    'processing_time_seconds': round(direct_time, 2)
                },
                'method2_caption_based': {
                    'image1_caption': caption_result['caption1'],
                    'image2_caption': caption_result['caption2'],
                    'modification_text': caption_result['comparison'],
                    'processing_time_seconds': round(caption_time, 2)
                },
                'method3_cot_single_call': {
                    'image1_caption': cot_result['image1_caption'],
                    'image2_caption': cot_result['image2_caption'],
                    'difference_list': cot_result['difference_list'],
                    'modification_text': cot_result['modification_text'],
                    'full_response': cot_result['full_response'],
                    'processing_time_seconds': round(cot_time, 2)
                }
            }
            
        except Exception as e:
            return {
                'method1_direct': {
                    'modification_text': f"Error in direct comparison: {str(e)}",
                    'processing_time_seconds': 0
                },
                'method2_caption_based': {
                    'image1_caption': f"Error: {str(e)}",
                    'image2_caption': f"Error: {str(e)}",
                    'modification_text': f"Error in caption-based comparison: {str(e)}",
                    'processing_time_seconds': 0
                },
                'method3_cot_single_call': {
                    'image1_caption': f"Error: {str(e)}",
                    'image2_caption': f"Error: {str(e)}",
                    'difference_list': f"Error: {str(e)}",
                    'modification_text': f"Error in COT comparison: {str(e)}",
                    'full_response': f"Error in COT comparison: {str(e)}",
                    'processing_time_seconds': 0
                }
            }
    
    def test_image_group(self, group_name: str, image_list: List[str]) -> List[Dict]:
        """Test image comparisons within a group"""
        print(f"\nTesting group: {group_name} (contains {len(image_list)} images)")
        
        results = []
        comparisons = []
        
        # Generate all possible image pairs
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                comparisons.append((image_list[i], image_list[j]))
        
        # If too many pairs, randomly select a subset
        if len(comparisons) > self.args.max_pairs_per_group:
            comparisons = random.sample(comparisons, self.args.max_pairs_per_group)
        
        print(f"Will compare {len(comparisons)} image pairs")
        
        for idx, (img1_name, img2_name) in enumerate(comparisons, 1):
            print(f"  Comparison {idx}/{len(comparisons)}: {img1_name} vs {img2_name}")
            
            start_time = time.time()
            comparison_results = self.compare_two_images(img1_name, img2_name)
            total_processing_time = time.time() - start_time
            
            result = {
                'group_name': group_name,
                'image1_id': img1_name,
                'image2_id': img2_name,
                'comparison_methods': comparison_results,
                'total_processing_time_seconds': round(total_processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            print(f"    Total processing time: {total_processing_time:.2f}s")
            
            # Display results for all three methods
            print(f"    Method 1 (Direct): {comparison_results['method1_direct']['modification_text'][:100]}...")
            print(f"    Method 2 (Caption): {comparison_results['method2_caption_based']['modification_text'][:100]}...")
            print(f"    Method 3 (COT): {comparison_results['method3_cot_single_call']['modification_text'][:100]}...")
            print(f"      └─ Image1 Caption: {comparison_results['method3_cot_single_call']['image1_caption'][:80]}...")
            print(f"      └─ Image2 Caption: {comparison_results['method3_cot_single_call']['image2_caption'][:80]}...")
        
        return results
    
    def run_test(self):
        """Run complete test"""
        print(f"\nStarting CIRR Multi-Image Understanding Test")
        print(f"Will test {min(self.args.num_groups, len(self.image_groups))} groups")
        print("=" * 60)
        
        # Randomly select groups to test
        selected_groups = dict(random.sample(
            list(self.image_groups.items()), 
            min(self.args.num_groups, len(self.image_groups))
        ))
        
        all_results = []
        test_metadata = {
            'test_start_time': datetime.now().isoformat(),
            'model_name': self.args.model_name,
            'device': self.device,
            'num_groups_tested': len(selected_groups),
            'max_pairs_per_group': self.args.max_pairs_per_group,
            'random_seed': self.args.seed,
            'comparison_methods': {
                'method1': 'Direct image comparison',
                'method2': 'Caption-based comparison',
                'method3': 'COT single-call comparison'
            },
            'parameters': vars(self.args)
        }
        
        for group_idx, (group_name, image_list) in enumerate(selected_groups.items(), 1):
            print(f"\n[{group_idx}/{len(selected_groups)}] Processing group: {group_name}")
            
            try:
                group_results = self.test_image_group(group_name, image_list)
                all_results.extend(group_results)
                
                print(f"✓ Group {group_name} completed, generated {len(group_results)} comparison results")
                
            except Exception as e:
                print(f"✗ Group {group_name} processing failed: {e}")
                continue
        
        # Save results
        test_metadata['test_end_time'] = datetime.now().isoformat()
        test_metadata['total_comparisons'] = len(all_results)
        
        final_results = {
            'metadata': test_metadata,
            'results': all_results
        }
        
        self.save_results(final_results)
        self.print_summary(final_results)
    
    def save_results(self, results: Dict):
        """Save test results"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {self.output_file}")
    
    def print_summary(self, results: Dict):
        """Print test summary"""
        metadata = results['metadata']
        comparisons = results['results']
        
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Model: {metadata['model_name']}")
        print(f"Device: {metadata['device']}")
        print(f"Groups tested: {metadata['num_groups_tested']}")
        print(f"Total comparisons: {metadata['total_comparisons']}")
        print(f"Test start time: {metadata['test_start_time']}")
        print(f"Test end time: {metadata['test_end_time']}")
        print(f"Comparison methods: Direct + Caption-based + COT")
        
        if comparisons:
            avg_time = sum(r['total_processing_time_seconds'] for r in comparisons) / len(comparisons)
            print(f"Average processing time: {avg_time:.2f}s per comparison pair")
            
            # Calculate method-specific times
            method1_times = [r['comparison_methods']['method1_direct']['processing_time_seconds'] for r in comparisons]
            method2_times = [r['comparison_methods']['method2_caption_based']['processing_time_seconds'] for r in comparisons]
            method3_times = [r['comparison_methods']['method3_cot_single_call']['processing_time_seconds'] for r in comparisons]
            
            avg_method1 = sum(method1_times) / len(method1_times)
            avg_method2 = sum(method2_times) / len(method2_times)
            avg_method3 = sum(method3_times) / len(method3_times)
            
            print(f"Average time - Method 1 (Direct): {avg_method1:.2f}s")
            print(f"Average time - Method 2 (Caption-based): {avg_method2:.2f}s")
            print(f"Average time - Method 3 (COT): {avg_method3:.2f}s")
        
        print(f"\nDetailed results saved to: {self.output_file}")
        print("=" * 60)


def main():
    args = parse_args()
    
    print("CIRR Multi-Image Understanding Test")
    print("=" * 60)
    print(f"CIRR data directory: {args.cirr_data_dir}")
    print(f"CIRR image directory: {args.cirr_image_dir}")
    print(f"Model: {args.model_name}")
    print(f"Groups to test: {args.num_groups}")
    print(f"Max pairs per group: {args.max_pairs_per_group}")
    print(f"Comparison methods: Direct + Caption-based + COT")
    
    try:
        tester = CIRRMultiImageTester(args)
        tester.run_test()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise


if __name__ == "__main__":
    main()
