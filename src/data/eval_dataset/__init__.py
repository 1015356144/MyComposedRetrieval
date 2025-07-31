# Basic evaluation datasets for image tasks
from .image_cls_dataset import load_image_cls_dataset
from .image_qa_dataset import load_image_qa_dataset
from .image_t2i_eval import load_image_t2i_dataset
from .image_i2t_eval import load_image_i2t_dataset
from .image_i2i_vg_dataset import load_image_i2i_vg_dataset

# Document and visual reasoning
from .vidore_dataset import load_vidore_dataset
from .visrag_dataset import load_visrag_dataset

__all__ = [
    'load_image_cls_dataset',
    'load_image_qa_dataset', 
    'load_image_t2i_dataset',
    'load_image_i2t_dataset',
    'load_image_i2i_vg_dataset',
    'load_vidore_dataset',
    'load_visrag_dataset',
]

# VisDoc
from .vidore_dataset import load_vidore_dataset
from .visrag_dataset import load_visrag_dataset
