"""
模型注册表
作用：
    统一管理所有可用模型，通过字符串名称获取模型类
主要功能：
    - get_model(name, **kwargs): 根据名称创建模型实例
    - list_models(): 列出所有可用模型
依赖：
    torch
注意：
    所有模型必须实现统一的 predict() 接口
示例：
    from models import get_model
    model = get_model('resnet_unet', pretrained=True)
    action = model.predict(image, state)
"""

from typing import Dict, Callable, List
import torch.nn as nn

# 模型注册表：name → (module_path, class_name)
REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    模型注册装饰器

    Args:
        name: 模型名称

    示例:
        @register_model('my_model')
        class MyModel(nn.Module):
            ...
    """
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """
    根据名称创建模型实例

    Args:
        name: 模型名称（如 'resnet_unet', 'toge'）
        **kwargs: 传递给模型构造函数的参数

    Returns:
        模型实例

    Raises:
        ValueError: 如果模型名称不存在
    """
    if name not in REGISTRY:
        available = ', '.join(list_models())
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )

    model_cls = REGISTRY[name]
    return model_cls(**kwargs)


def list_models() -> List[str]:
    """
    列出所有可用模型

    Returns:
        模型名称列表
    """
    return list(REGISTRY.keys())


# 导入并注册所有模型
from models.resnet18_unet import FPVDiffusionPolicy
from models.toge_policy import TOGEPolicy

# 注册模型
REGISTRY['resnet_unet'] = FPVDiffusionPolicy
REGISTRY['fpv_diffusion'] = FPVDiffusionPolicy  # 别名
REGISTRY['toge'] = TOGEPolicy


__all__ = ['get_model', 'list_models', 'register_model', 'REGISTRY']
