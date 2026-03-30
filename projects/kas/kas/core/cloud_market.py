"""
KAS Cloud Market Client
云端市场 API 客户端
"""
import os
import json
import requests
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from kas.core.models import Agent
from kas.core.config import get_config


@dataclass
class CloudPackageInfo:
    """云端包信息"""
    id: int
    name: str
    version: str
    description: Optional[str]
    author_name: str
    downloads: int
    rating: float
    tags: List[str]
    created_at: str


class CloudMarketClient:
    """云端市场客户端"""
    
    DEFAULT_API_URL = "http://localhost:8000"
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        初始化云端市场客户端
        
        Args:
            api_url: API 基础 URL，默认从配置读取
            api_key: API 认证密钥
        """
        # 从配置读取
        config = get_config()
        self.api_url = api_url or config.cloud.api_url or self.DEFAULT_API_URL
        self.api_url = self.api_url.rstrip('/')
        
        # API Key 从凭证读取（云端市场专用）
        self.api_key = api_key or config.credentials.get('cloud_api_key')
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """发送请求"""
        url = f"{self.api_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise CloudMarketError("无法连接到云端市场服务器，请检查:")
        except requests.exceptions.Timeout:
            raise CloudMarketError("请求超时，请稍后重试")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise CloudMarketError("认证失败，请检查 API Key")
            elif response.status_code == 404:
                raise CloudMarketError("资源不存在")
            elif response.status_code == 409:
                raise CloudMarketError(response.json().get('detail', '资源冲突'))
            else:
                raise CloudMarketError(f"请求失败: {e}")
        except Exception as e:
            raise CloudMarketError(f"请求错误: {e}")
    
    def is_available(self) -> bool:
        """检查云端是否可用"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def search(self, query: str = "", tags: Optional[List[str]] = None,
               limit: int = 20) -> List[CloudPackageInfo]:
        """
        搜索云端 Agent
        
        Args:
            query: 搜索关键词
            tags: 标签过滤
            limit: 返回数量限制
            
        Returns:
            包信息列表
        """
        params = {'q': query, 'limit': limit}
        if tags:
            params['tags'] = tags
        
        results = self._request('GET', '/api/v1/market/search', params=params)
        
        return [CloudPackageInfo(
            id=r['id'],
            name=r['name'],
            version=r['version'],
            description=r.get('description'),
            author_name=r['author_name'],
            downloads=r['downloads'],
            rating=r['rating'],
            tags=r.get('tags', []),
            created_at=r['created_at']
        ) for r in results]
    
    def get_package_info(self, package_id: int) -> Dict:
        """获取包详情"""
        return self._request('GET', f'/api/v1/market/packages/{package_id}')
    
    def publish(self, package_path: str, name: str, version: str,
                description: Optional[str] = None, tags: List[str] = None) -> Dict:
        """
        发布包到云端市场
        
        Args:
            package_path: .kas-agent 文件路径
            name: 包名称
            version: 版本号
            description: 描述
            tags: 标签列表
            
        Returns:
            发布结果
        """
        if not self.api_key:
            raise CloudMarketError("发布需要 API Key，请先运行 'kas config-setup' 配置")
        
        # 准备元数据
        data = {
            'name': name,
            'version': version,
            'description': description or '',
            'tags': tags or [],
            'capabilities': []
        }
        
        # 上传文件
        with open(package_path, 'rb') as f:
            files = {'file': (f"{name}-{version}.kas-agent", f, 'application/octet-stream')}
            response = self._request(
                'POST',
                '/api/v1/market/publish',
                data={'package': json.dumps(data)},
                files=files
            )
        
        return response
    
    def download(self, package_id: int, output_dir: Optional[str] = None) -> str:
        """
        下载包
        
        Args:
            package_id: 包 ID
            output_dir: 输出目录，默认当前目录
            
        Returns:
            下载文件路径
        """
        url = f"{self.api_url}/api/v1/market/packages/{package_id}/download"
        
        try:
            response = self.session.post(url, timeout=60)
            response.raise_for_status()
            
            # 获取文件名
            content_disposition = response.headers.get('content-disposition', '')
            filename = f"package-{package_id}.kas-agent"
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"\'')
            
            # 保存文件
            output_path = Path(output_dir or '.') / filename
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return str(output_path)
            
        except Exception as e:
            raise CloudMarketError(f"下载失败: {e}")
    
    def rate(self, package_id: int, score: int, comment: Optional[str] = None) -> Dict:
        """
        给包评分
        
        Args:
            package_id: 包 ID
            score: 评分 1-5
            comment: 评论
            
        Returns:
            评分结果
        """
        if not self.api_key:
            raise CloudMarketError("评分需要登录，请先配置 API Key")
        
        data = {'score': score}
        if comment:
            data['comment'] = comment
        
        return self._request(
            'POST',
            f'/api/v1/market/packages/{package_id}/rate',
            json=data
        )


class CloudMarketError(Exception):
    """云端市场错误"""
    pass


def get_cloud_client() -> CloudMarketClient:
    """获取云端市场客户端"""
    return CloudMarketClient()
