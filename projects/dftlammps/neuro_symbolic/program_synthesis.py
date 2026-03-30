"""
程序合成模块 - 从示例学习程序与代码生成

实现基于神经网络的程序合成系统，支持从输入-输出示例学习程序、
代码补全与生成、以及自动验证与测试。
"""

from typing import List, Dict, Tuple, Optional, Callable, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import re
import ast
from typing import get_type_hints


class ASTNodeType(Enum):
    """AST节点类型"""
    CONSTANT = auto()
    VARIABLE = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    CALL = auto()
    IF = auto()
    LOOP = auto()
    ASSIGN = auto()
    SEQUENCE = auto()
    FUNCTION_DEF = auto()


@dataclass
class ASTNode:
    """抽象语法树节点"""
    node_type: ASTNodeType
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        result = f"{indent_str}{self.node_type.name}"
        if self.value is not None:
            result += f": {self.value}"
        if self.attributes:
            result += f" {self.attributes}"
        result += "\n"
        for child in self.children:
            result += child.__repr__(indent + 1)
        return result
    
    def to_code(self) -> str:
        """转换为可执行代码"""
        return ASTToCodeConverter().convert(self)


class ASTToCodeConverter:
    """AST到代码转换器"""
    
    def convert(self, node: ASTNode) -> str:
        """将AST转换为Python代码"""
        method = getattr(self, f'_convert_{node.node_type.name.lower()}', 
                        self._convert_default)
        return method(node)
    
    def _convert_constant(self, node: ASTNode) -> str:
        return repr(node.value)
    
    def _convert_variable(self, node: ASTNode) -> str:
        return str(node.value)
    
    def _convert_binary_op(self, node: ASTNode) -> str:
        op_map = {
            '+': '+', '-': '-', '*': '*', '/': '/',
            '//': '//', '%': '%', '**': '**',
            '<': '<', '>': '>', '<=': '<=', '>=': '>=',
            '==': '==', '!=': '!=', 'and': 'and', 'or': 'or'
        }
        op = op_map.get(node.value, node.value)
        left = self.convert(node.children[0])
        right = self.convert(node.children[1])
        return f"({left} {op} {right})"
    
    def _convert_unary_op(self, node: ASTNode) -> str:
        op_map = {'-': '-', 'not': 'not', '~': '~'}
        op = op_map.get(node.value, node.value)
        operand = self.convert(node.children[0])
        return f"({op}{operand})"
    
    def _convert_call(self, node: ASTNode) -> str:
        func_name = node.value
        args = [self.convert(child) for child in node.children]
        return f"{func_name}({', '.join(args)})"
    
    def _convert_if(self, node: ASTNode) -> str:
        condition = self.convert(node.children[0])
        then_branch = self.convert(node.children[1])
        if len(node.children) > 2:
            else_branch = self.convert(node.children[2])
            return f"if {condition}:\n    {then_branch}\nelse:\n    {else_branch}"
        return f"if {condition}:\n    {then_branch}"
    
    def _convert_loop(self, node: ASTNode) -> str:
        loop_type = node.attributes.get('loop_type', 'for')
        if loop_type == 'for':
            var = node.children[0].value
            iterable = self.convert(node.children[1])
            body = self.convert(node.children[2])
            return f"for {var} in {iterable}:\n    {body}"
        else:  # while
            condition = self.convert(node.children[0])
            body = self.convert(node.children[1])
            return f"while {condition}:\n    {body}"
    
    def _convert_assign(self, node: ASTNode) -> str:
        var = node.children[0].value
        value = self.convert(node.children[1])
        return f"{var} = {value}"
    
    def _convert_sequence(self, node: ASTNode) -> str:
        statements = [self.convert(child) for child in node.children]
        return '\n'.join(statements)
    
    def _convert_function_def(self, node: ASTNode) -> str:
        name = node.value
        params = node.attributes.get('params', [])
        body = self.convert(node.children[0])
        param_str = ', '.join(params)
        return f"def {name}({param_str}):\n    {body.replace(chr(10), chr(10)+'    ')}"
    
    def _convert_default(self, node: ASTNode) -> str:
        return f"# Unknown node type: {node.node_type}"


class NeuralProgramSynthesizer(nn.Module):
    """
    神经程序合成器
    
    基于Transformer的程序合成模型，从输入-输出示例学习程序。
    """
    
    def __init__(self,
                 vocab_size: int = 1000,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_program_length: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_program_length = max_program_length
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_program_length, embedding_dim)
        
        # Transformer编码器（编码输入-输出示例）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.example_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 示例聚合
        self.example_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Transformer解码器（生成程序）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.program_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # 输出生成
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # 程序验证网络
        self.verification_net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def encode_examples(self, 
                       examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        编码输入-输出示例
        
        Args:
            examples: List of (input, output) pairs
        
        Returns:
            example_encoding: (1, embedding_dim) 聚合的示例表示
        """
        encoded_examples = []
        
        for inp, out in examples:
            # 展平并转换为token索引
            inp_flat = inp.flatten().long() % self.vocab_size
            out_flat = out.flatten().long() % self.vocab_size
            
            # 组合输入和输出
            combined = torch.cat([inp_flat, out_flat])
            
            # 截断或填充到固定长度
            if len(combined) > self.max_program_length:
                combined = combined[:self.max_program_length]
            else:
                padding = torch.zeros(self.max_program_length - len(combined))
                combined = torch.cat([combined, padding])
            
            # 嵌入
            positions = torch.arange(len(combined))
            embedded = self.token_embedding(combined.long()) + \
                      self.position_embedding(positions)
            
            encoded_examples.append(embedded.unsqueeze(0))
        
        # 编码所有示例
        if encoded_examples:
            batch_encoded = torch.cat(encoded_examples, dim=0)
            encoded = self.example_encoder(batch_encoded)
            
            # 聚合（平均池化）
            aggregated = encoded.mean(dim=1)
            final_encoding = self.example_aggregator(aggregated)
            
            return final_encoding.mean(dim=0, keepdim=True).unsqueeze(0)
        
        return torch.zeros(1, 1, self.embedding_dim)
    
    def forward(self,
                examples: List[Tuple[torch.Tensor, torch.Tensor]],
                target_program: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            examples: 输入-输出示例列表
            target_program: 目标程序token序列（训练时使用）
        
        Returns:
            logits: (1, seq_len, vocab_size) 程序token logits
        """
        # 编码示例
        memory = self.encode_examples(examples)
        
        # 准备解码器输入
        if target_program is not None:
            # 训练模式：使用teacher forcing
            decoder_input = target_program
        else:
            # 推理模式：使用BOS token开始
            decoder_input = torch.zeros(1, 1, dtype=torch.long)
        
        # 嵌入解码器输入
        positions = torch.arange(decoder_input.shape[1])
        decoder_embedded = self.token_embedding(decoder_input) + \
                          self.position_embedding(positions)
        
        # 生成目标掩码
        tgt_mask = self._generate_square_subsequent_mask(decoder_input.shape[1])
        
        # 解码
        decoded = self.program_decoder(
            decoder_embedded, memory, tgt_mask=tgt_mask
        )
        
        # 投影到词汇表
        logits = self.output_projection(decoded)
        
        return logits
    
    def synthesize(self,
                  examples: List[Tuple[torch.Tensor, torch.Tensor]],
                  max_length: int = 50,
                  temperature: float = 1.0,
                  beam_width: int = 5) -> Tuple[List[int], float]:
        """
        从示例合成程序
        
        Args:
            examples: 输入-输出示例
            max_length: 最大程序长度
            temperature: 采样温度
            beam_width: 束搜索宽度
        
        Returns:
            program_tokens: 合成的程序token序列
            confidence: 置信度分数
        """
        # 编码示例
        memory = self.encode_examples(examples)
        
        # 束搜索
        beams = [(torch.zeros(1, 1, dtype=torch.long), 0.0)]
        completed = []
        
        for step in range(max_length):
            new_beams = []
            
            for tokens, score in beams:
                if tokens[0, -1].item() == 2:  # EOS token
                    completed.append((tokens, score))
                    continue
                
                # 嵌入当前序列
                positions = torch.arange(tokens.shape[1])
                embedded = self.token_embedding(tokens) + \
                          self.position_embedding(positions)
                
                # 解码
                tgt_mask = self._generate_square_subsequent_mask(tokens.shape[1])
                decoded = self.program_decoder(embedded, memory, tgt_mask=tgt_mask)
                logits = self.output_projection(decoded[:, -1, :])
                
                # 采样top-k
                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_width)
                
                for prob, idx in zip(topk_probs[0], topk_indices[0]):
                    new_tokens = torch.cat([tokens, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + torch.log(prob).item()
                    new_beams.append((new_tokens, new_score))
            
            # 选择top beam_width个
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
            
            if len(completed) >= beam_width:
                break
        
        # 返回最佳结果
        if completed:
            best_tokens, best_score = max(completed, key=lambda x: x[1])
            confidence = np.exp(best_score / best_tokens.shape[1])
            return best_tokens[0].tolist(), confidence
        
        if beams:
            best_tokens, best_score = beams[0]
            confidence = np.exp(best_score / best_tokens.shape[1])
            return best_tokens[0].tolist(), confidence
        
        return [], 0.0
    
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """生成三角掩码"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class CodeCompletionModel(nn.Module):
    """
    代码补全模型
    
    基于大规模预训练的代码补全系统。
    """
    
    def __init__(self,
                 vocab_size: int = 50000,
                 embedding_dim: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 max_context_length: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_context_length, embedding_dim)
        
        # Transformer
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                context: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            context: (batch_size, seq_len) 上下文token
            target: (batch_size, target_len) 目标token（训练时使用）
        
        Returns:
            logits: 下一个token的预测logits
        """
        batch_size, seq_len = context.shape
        
        # 嵌入
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        embedded = self.token_embedding(context) + self.position_embedding(positions)
        embedded = self.dropout(embedded)
        
        # 因果掩码
        mask = self._generate_causal_mask(seq_len)
        
        # Transformer
        output = self.transformer(embedded, embedded, tgt_mask=mask)
        
        # 预测下一个token
        logits = self.output_layer(output)
        
        return logits
    
    def complete(self,
                context: str,
                tokenizer: Any,
                max_new_tokens: int = 100,
                temperature: float = 0.8,
                top_p: float = 0.95) -> str:
        """
        代码补全
        
        Args:
            context: 代码上下文
            tokenizer: 分词器
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus sampling参数
        
        Returns:
            completed_code: 补全后的代码
        """
        # 编码上下文
        context_tokens = tokenizer.encode(context)
        input_ids = torch.tensor([context_tokens])
        
        generated = []
        
        for _ in range(max_new_tokens):
            # 截断到最大长度
            if input_ids.shape[1] > self.max_context_length:
                input_ids = input_ids[:, -self.max_context_length:]
            
            # 前向传播
            with torch.no_grad():
                logits = self.forward(input_ids)
            
            # 采样下一个token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-p采样
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token.item())
            
            # 检查是否生成结束标记
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 更新输入
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # 解码生成结果
        completed_code = tokenizer.decode(context_tokens + generated)
        return completed_code
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class ProgramVerifier:
    """
    程序验证器
    
    自动验证合成程序的正确性。
    """
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        
        # 安全执行环境
        self.safe_globals = {
            '__builtins__': {
                'True': True, 'False': False, 'None': None,
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'int': int, 'float': float, 'str': str, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set,
                'print': print, 'sorted': sorted, 'reversed': reversed,
                'all': all, 'any': any
            }
        }
    
    def verify(self, 
              program_code: str,
              test_cases: List[Tuple[Any, Any]]) -> Tuple[bool, float, List[str]]:
        """
        验证程序
        
        Args:
            program_code: 程序代码字符串
            test_cases: (输入, 期望输出)测试用例列表
        
        Returns:
            passed: 是否通过所有测试
            score: 通过率
            errors: 错误信息列表
        """
        errors = []
        passed_count = 0
        
        try:
            # 解析语法
            ast.parse(program_code)
        except SyntaxError as e:
            return False, 0.0, [f"Syntax error: {e}"]
        
        # 执行每个测试用例
        for i, (test_input, expected_output) in enumerate(test_cases):
            try:
                result = self._execute_with_timeout(
                    program_code, test_input, expected_output
                )
                
                if result == expected_output:
                    passed_count += 1
                else:
                    errors.append(
                        f"Test {i}: Expected {expected_output}, got {result}"
                    )
            except Exception as e:
                errors.append(f"Test {i}: Runtime error - {e}")
        
        score = passed_count / len(test_cases) if test_cases else 0.0
        return passed_count == len(test_cases), score, errors
    
    def _execute_with_timeout(self,
                             code: str,
                             test_input: Any,
                             expected_output: Any) -> Any:
        """在受限环境中执行代码"""
        # 创建执行环境
        local_ns = {}
        
        # 添加输入到命名空间
        if isinstance(test_input, dict):
            local_ns.update(test_input)
        else:
            local_ns['input_val'] = test_input
        
        # 执行代码
        exec(code, self.safe_globals, local_ns)
        
        # 获取结果（假设函数名为'solve'）
        if 'solve' in local_ns:
            if isinstance(test_input, dict):
                return local_ns['solve'](**test_input)
            else:
                return local_ns['solve'](test_input)
        
        # 或者返回最后一个变量的值
        return local_ns.get('result', None)
    
    def generate_test_cases(self,
                           specification: str,
                           num_cases: int = 10) -> List[Tuple[Any, Any]]:
        """
        根据规范生成测试用例
        """
        test_cases = []
        
        # 解析规范（简化实现）
        # 实际应用中可以使用LLM或更复杂的方法
        
        # 生成边界值测试
        if "array" in specification.lower() or "list" in specification.lower():
            test_cases.extend([
                ([], []),  # 空数组
                ([1], [1]),  # 单元素
                ([1, 2, 3], [1, 2, 3]),  # 正常数组
                ([3, 2, 1], [1, 2, 3]),  # 逆序
            ])
        
        if "number" in specification.lower() or "int" in specification.lower():
            test_cases.extend([
                (0, 0),  # 零
                (1, 1),  # 正数
                (-1, -1),  # 负数
                (1000000, 1000000),  # 大数
            ])
        
        return test_cases[:num_cases]


class SearchBasedSynthesizer:
    """
    基于搜索的程序合成器
    
    使用枚举搜索和遗传算法合成程序。
    """
    
    def __init__(self,
                 primitives: List[Callable],
                 max_program_depth: int = 5,
                 population_size: int = 100,
                 mutation_rate: float = 0.1):
        self.primitives = primitives
        self.max_program_depth = max_program_depth
        self.population_size = population_size
        self.mutation_rate = mutation_rate
    
    def enumerate_search(self,
                        examples: List[Tuple[Any, Any]],
                        max_size: int = 1000) -> Optional[Callable]:
        """
        枚举搜索
        
        系统地枚举所有可能的程序直到找到满足示例的程序。
        """
        from itertools import product
        
        for depth in range(1, self.max_program_depth + 1):
            # 生成所有可能的程序组合
            for combination in product(self.primitives, repeat=depth):
                # 构建程序
                def program(x):
                    result = x
                    for func in combination:
                        try:
                            result = func(result)
                        except:
                            break
                    return result
                
                # 测试程序
                if self._test_program(program, examples):
                    return program
        
        return None
    
    def genetic_search(self,
                      examples: List[Tuple[Any, Any]],
                      generations: int = 100) -> Optional[Callable]:
        """
        遗传算法搜索
        
        使用进化算法搜索最优程序。
        """
        # 初始化种群
        population = self._initialize_population()
        
        for generation in range(generations):
            # 评估适应度
            fitness_scores = [
                self._evaluate_fitness(individual, examples)
                for individual in population
            ]
            
            # 检查是否找到解
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] == 1.0:
                return population[best_idx]
            
            # 选择
            selected = self._select(population, fitness_scores)
            
            # 交叉和变异
            population = self._evolve(selected)
        
        # 返回最佳个体
        fitness_scores = [
            self._evaluate_fitness(individual, examples)
            for individual in population
        ]
        best_idx = np.argmax(fitness_scores)
        return population[best_idx] if fitness_scores[best_idx] > 0.5 else None
    
    def _initialize_population(self) -> List[List[Callable]]:
        """初始化种群"""
        import random
        population = []
        for _ in range(self.population_size):
            length = random.randint(1, self.max_program_depth)
            individual = [random.choice(self.primitives) for _ in range(length)]
            population.append(individual)
        return population
    
    def _evaluate_fitness(self,
                         individual: List[Callable],
                         examples: List[Tuple[Any, Any]]) -> float:
        """评估适应度"""
        def program(x):
            result = x
            for func in individual:
                try:
                    result = func(result)
                except:
                    return None
            return result
        
        correct = 0
        for inp, expected in examples:
            try:
                if program(inp) == expected:
                    correct += 1
            except:
                pass
        
        return correct / len(examples)
    
    def _select(self,
               population: List[List[Callable]],
               fitness_scores: List[float]) -> List[List[Callable]]:
        """选择操作（锦标赛选择）"""
        import random
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def _evolve(self, selected: List[List[Callable]]) -> List[List[Callable]]:
        """进化操作（交叉和变异）"""
        import random
        new_population = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]
            
            # 交叉
            if random.random() < 0.7:
                crossover_point = random.randint(1, min(len(parent1), len(parent2)))
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _mutate(self, individual: List[Callable]) -> List[Callable]:
        """变异操作"""
        import random
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'replace'])
            
            if mutation_type == 'add' and len(individual) < self.max_program_depth:
                individual.append(random.choice(self.primitives))
            elif mutation_type == 'remove' and len(individual) > 1:
                individual.pop(random.randint(0, len(individual) - 1))
            elif mutation_type == 'replace' and individual:
                idx = random.randint(0, len(individual) - 1)
                individual[idx] = random.choice(self.primitives)
        
        return individual
    
    def _test_program(self, program: Callable, examples: List[Tuple[Any, Any]]) -> bool:
        """测试程序"""
        for inp, expected in examples:
            try:
                if program(inp) != expected:
                    return False
            except:
                return False
        return True


# ==================== DSL（领域特定语言） ====================

class MaterialDSL:
    """材料科学领域特定语言"""
    
    @staticmethod
    def filter_by_property(materials: List[Dict], property_name: str, 
                          min_val: float, max_val: float) -> List[Dict]:
        """按属性过滤材料"""
        return [
            m for m in materials
            if min_val <= m.get(property_name, float('inf')) <= max_val
        ]
    
    @staticmethod
    def sort_by_property(materials: List[Dict], property_name: str,
                        reverse: bool = False) -> List[Dict]:
        """按属性排序材料"""
        return sorted(materials, key=lambda m: m.get(property_name, 0), 
                    reverse=reverse)
    
    @staticmethod
    def compute_average(materials: List[Dict], property_name: str) -> float:
        """计算属性平均值"""
        values = [m.get(property_name, 0) for m in materials if property_name in m]
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def find_similar(materials: List[Dict], target: Dict,
                    properties: List[str]) -> List[Dict]:
        """找到相似材料"""
        def similarity(m1, m2):
            score = 0
            for prop in properties:
                if prop in m1 and prop in m2:
                    v1, v2 = m1[prop], m2[prop]
                    score += 1 - abs(v1 - v2) / max(abs(v1) + abs(v2), 1e-8)
            return score / len(properties) if properties else 0
        
        return sorted(materials, key=lambda m: similarity(m, target), reverse=True)
    
    @staticmethod
    def predict_property(materials: List[Dict], target_property: str,
                        feature_properties: List[str]) -> Dict:
        """基于其他属性预测属性（简化实现）"""
        # 使用简单线性回归
        from sklearn.linear_model import LinearRegression
        
        X = [[m.get(p, 0) for p in feature_properties] for m in materials 
             if target_property in m]
        y = [m[target_property] for m in materials if target_property in m]
        
        if len(X) > 1:
            model = LinearRegression().fit(X, y)
            
            # 为缺失该属性的材料预测
            for m in materials:
                if target_property not in m:
                    features = [[m.get(p, 0) for p in feature_properties]]
                    m[f'predicted_{target_property}'] = model.predict(features)[0]
        
        return materials


# ==================== 实用函数 ====================

def build_ast_from_code(code: str) -> ASTNode:
    """从Python代码构建AST"""
    
    def convert_python_ast(node) -> ASTNode:
        """递归转换Python AST"""
        if isinstance(node, ast.Constant):
            return ASTNode(ASTNodeType.CONSTANT, node.value)
        
        elif isinstance(node, ast.Name):
            return ASTNode(ASTNodeType.VARIABLE, node.id)
        
        elif isinstance(node, ast.BinOp):
            op_map = {
                ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
                ast.FloorDiv: '//', ast.Mod: '%', ast.Pow: '**'
            }
            op = op_map.get(type(node.op), '?')
            return ASTNode(ASTNodeType.BINARY_OP, op, [
                convert_python_ast(node.left),
                convert_python_ast(node.right)
            ])
        
        elif isinstance(node, ast.UnaryOp):
            op_map = {ast.UAdd: '+', ast.USub: '-', ast.Not: 'not', ast.Invert: '~'}
            op = op_map.get(type(node.op), '?')
            return ASTNode(ASTNodeType.UNARY_OP, op, [convert_python_ast(node.operand)])
        
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            args = [convert_python_ast(arg) for arg in node.args]
            return ASTNode(ASTNodeType.CALL, func_name, args)
        
        elif isinstance(node, ast.If):
            test = convert_python_ast(node.test)
            body = ASTNode(ASTNodeType.SEQUENCE, children=[
                convert_python_ast(stmt) for stmt in node.body
            ])
            orelse = None
            if node.orelse:
                orelse = ASTNode(ASTNodeType.SEQUENCE, children=[
                    convert_python_ast(stmt) for stmt in node.orelse
                ])
            children = [test, body, orelse] if orelse else [test, body]
            return ASTNode(ASTNodeType.IF, children=children)
        
        elif isinstance(node, ast.For):
            target = convert_python_ast(node.target)
            iter_node = convert_python_ast(node.iter)
            body = ASTNode(ASTNodeType.SEQUENCE, children=[
                convert_python_ast(stmt) for stmt in node.body
            ])
            return ASTNode(ASTNodeType.LOOP, children=[target, iter_node, body],
                         attributes={'loop_type': 'for'})
        
        elif isinstance(node, ast.While):
            test = convert_python_ast(node.test)
            body = ASTNode(ASTNodeType.SEQUENCE, children=[
                convert_python_ast(stmt) for stmt in node.body
            ])
            return ASTNode(ASTNodeType.LOOP, children=[test, body],
                         attributes={'loop_type': 'while'})
        
        elif isinstance(node, ast.Assign):
            targets = [convert_python_ast(t) for t in node.targets]
            value = convert_python_ast(node.value)
            return ASTNode(ASTNodeType.ASSIGN, children=[targets[0], value])
        
        elif isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            body = ASTNode(ASTNodeType.SEQUENCE, children=[
                convert_python_ast(stmt) for stmt in node.body
            ])
            return ASTNode(ASTNodeType.FUNCTION_DEF, node.name, [body],
                         attributes={'params': params})
        
        elif isinstance(node, ast.Module):
            return ASTNode(ASTNodeType.SEQUENCE, children=[
                convert_python_ast(stmt) for stmt in node.body
            ])
        
        else:
            # 默认处理：尝试处理子节点
            children = []
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    children.append(convert_python_ast(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            children.append(convert_python_ast(item))
            return ASTNode(ASTNodeType.SEQUENCE, children=children)
    
    tree = ast.parse(code)
    return convert_python_ast(tree)


def synthesize_material_query(examples: List[Tuple[Dict, List[Dict]]]) -> str:
    """
    从示例合成材料查询程序
    
    Args:
        examples: (输入条件, 输出材料列表)示例列表
    
    Returns:
        合成的Python代码
    """
    # 分析示例找出共同模式
    patterns = []
    
    for conditions, results in examples:
        # 检查是否是过滤操作
        if 'property_range' in conditions:
            patterns.append('filter')
        # 检查是否是排序操作
        if 'sort_by' in conditions:
            patterns.append('sort')
        # 检查是否是预测操作
        if 'predict' in conditions:
            patterns.append('predict')
    
    # 根据模式生成代码
    if 'filter' in patterns:
        code = '''
def query_materials(materials, conditions):
    result = materials
    if 'property_range' in conditions:
        prop, (min_val, max_val) = conditions['property_range']
        result = [m for m in result if min_val <= m.get(prop, 0) <= max_val]
    if 'sort_by' in conditions:
        prop, reverse = conditions['sort_by']
        result = sorted(result, key=lambda m: m.get(prop, 0), reverse=reverse)
    return result
'''
    else:
        code = '''
def query_materials(materials, conditions):
    return materials
'''
    
    return code


if __name__ == "__main__":
    print("=" * 60)
    print("程序合成模块测试")
    print("=" * 60)
    
    # 测试1: AST构建和代码生成
    print("\n测试1: AST构建与代码生成")
    code = '''
def add(a, b):
    return a + b
'''
    ast_node = build_ast_from_code(code)
    print("AST结构:")
    print(ast_node)
    generated_code = ast_node.to_code()
    print(f"\n生成的代码:\n{generated_code}")
    
    # 测试2: 程序验证
    print("\n测试2: 程序验证")
    verifier = ProgramVerifier()
    test_program = '''
def solve(x):
    return x * 2
'''
    test_cases = [(1, 2), (2, 4), (3, 6), (5, 10)]
    passed, score, errors = verifier.verify(test_program, test_cases)
    print(f"测试通过: {passed}")
    print(f"通过率: {score:.2%}")
    if errors:
        print(f"错误: {errors}")
    
    # 测试3: 材料DSL
    print("\n测试3: 材料DSL")
    dsl = MaterialDSL()
    materials = [
        {'name': 'Si', 'band_gap': 1.12, 'conductivity': 0.001},
        {'name': 'Ge', 'band_gap': 0.67, 'conductivity': 0.002},
        {'name': 'Cu', 'band_gap': 0, 'conductivity': 100},
        {'name': 'GaAs', 'band_gap': 1.42, 'conductivity': 0.001},
    ]
    
    semiconductors = dsl.filter_by_property(materials, 'band_gap', 0.1, 4.0)
    print(f"半导体材料: {[m['name'] for m in semiconductors]}")
    
    sorted_materials = dsl.sort_by_property(materials, 'band_gap')
    print(f"按带隙排序: {[m['name'] for m in sorted_materials]}")
    
    avg_gap = dsl.compute_average(materials, 'band_gap')
    print(f"平均带隙: {avg_gap:.3f} eV")
    
    # 测试4: 程序合成器
    print("\n测试4: 神经程序合成器")
    synthesizer = NeuralProgramSynthesizer(
        vocab_size=100,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        max_program_length=20
    )
    
    # 创建示例数据
    examples = [
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])),
        (torch.tensor([2.0, 4.0]), torch.tensor([4.0, 8.0])),
    ]
    
    logits = synthesizer.forward(examples)
    print(f"输出logits形状: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
