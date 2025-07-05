import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PlotPipeline:
    def __init__(self):
        pass

    def run(self):
        # 执行位置编码可视化
        self.plot_sinusoidal_position()
        
        # 分析高频区域的编码冲突
        self.analyze_high_frequency_collision()
        
        # 分析正弦位置编码的优势
        self.analyze_sinusoidal_benefits()
        
        # 分析RoPE的优势
        self.analyze_rope_benefits()
        
        # 对比可视化
        self.visualize_rope_vs_sinusoidal()

    def create_positional_encoding(self, seq_len, d_model):
        """
        创建正弦位置编码
        Args:
            seq_len: 序列长度（token数量）
            d_model: 模型维度
        Returns:
            位置编码张量，形状为 (seq_len, d_model)
        """
        # 创建位置编码矩阵
        pe = torch.zeros(seq_len, d_model)
        
        # 创建位置索引
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # 创建维度索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        # 计算正弦和余弦值
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        
        return pe

    def plot_sinusoidal_position(self):
        """绘制位置编码的可视化图"""
        # 参数设置
        seq_len = 128  # token数量
        d_model = 512  # 模型维度
        
        # 生成位置编码
        pe = self.create_positional_encoding(seq_len, d_model)
        
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 绘制热力图
        plt.subplot(2, 2, 1)
        sns.heatmap(pe.numpy(), cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Positional Encoding Value'})
        plt.title('Complete Positional Encoding Heatmap')
        plt.xlabel('Dimension Index')
        plt.ylabel('Token Position')
        
        # 绘制前64个维度的详细视图
        plt.subplot(2, 2, 2)
        sns.heatmap(pe[:, :64].numpy(), cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Positional Encoding Value'})
        plt.title('First 64 Dimensions Positional Encoding')
        plt.xlabel('Dimension Index')
        plt.ylabel('Token Position')
        
        # 绘制特定token的编码值
        plt.subplot(2, 2, 3)
        tokens_to_show = [0, 10, 20, 30, 40, 50]
        for i, token_pos in enumerate(tokens_to_show):
            plt.plot(pe[token_pos, :100].numpy(), 
                    label=f'Token {token_pos}', alpha=0.7)
        plt.title('Encoding Values for Different Token Positions (First 100 Dims)')
        plt.xlabel('Dimension Index')
        plt.ylabel('Encoding Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制特定维度的编码值
        plt.subplot(2, 2, 4)
        dims_to_show = [0, 1, 10, 11, 50, 51]
        for dim in dims_to_show:
            plt.plot(pe[:, dim].numpy(), 
                    label=f'Dimension {dim}', alpha=0.7)
        plt.title('Encoding Values for Different Dimensions Across Tokens')
        plt.xlabel('Token Position')
        plt.ylabel('Encoding Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('positional_encoding_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印一些统计信息
        print(f"位置编码形状: {pe.shape}")
        print(f"最大值: {pe.max().item():.4f}")
        print(f"最小值: {pe.min().item():.4f}")
        print(f"均值: {pe.mean().item():.4f}")
        print(f"标准差: {pe.std().item():.4f}")
        
        # 额外分析：展示位置编码的周期性特征
        print("\n位置编码分析:")
        print("="*50)
        
        # 分析前几个维度的周期性
        for dim in [0, 1, 10, 11, 50, 51]:
            values = pe[:, dim].numpy()
            print(f"维度 {dim}: 最大值={values.max():.4f}, 最小值={values.min():.4f}")
            
        # 显示不同token位置的编码相似性
        print("\n不同token位置的编码相似性 (余弦相似度):")
        from torch.nn.functional import cosine_similarity
        pos_pairs = [(0, 1), (0, 10), (10, 20), (50, 60)]
        for pos1, pos2 in pos_pairs:
            sim = cosine_similarity(pe[pos1].unsqueeze(0), pe[pos2].unsqueeze(0))
            print(f"Token {pos1} vs Token {pos2}: {sim.item():.4f}")

    def analyze_high_frequency_collision(self):
        """分析高频区域的位置编码冲突现象"""
        seq_len = 128
        d_model = 512
        
        # 生成位置编码
        pe = self.create_positional_encoding(seq_len, d_model)
        
        print("高频区域位置编码分析:")
        print("="*60)
        
        # 分析前几个维度的周期性
        high_freq_dims = [0, 1, 2, 3, 4, 5]  # 高频维度
        
        for dim in high_freq_dims:
            values = pe[:, dim].numpy()
            
            # 计算这个维度的理论周期
            if dim % 2 == 0:  # 偶数维度
                div_term = np.exp((dim // 2) * (-np.log(10000.0) / d_model))
                period = 2 * np.pi / div_term
                print(f"维度 {dim} (sin): 理论周期 = {period:.2f}")
            else:  # 奇数维度
                div_term = np.exp((dim // 2) * (-np.log(10000.0) / d_model))
                period = 2 * np.pi / div_term
                print(f"维度 {dim} (cos): 理论周期 = {period:.2f}")
            
            # 寻找相近的编码值
            tolerance = 0.01  # 容忍度
            collisions = []
            
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if abs(values[i] - values[j]) < tolerance:
                        collisions.append((i, j, values[i], values[j]))
            
            if collisions:
                print(f"  发现 {len(collisions)} 对相近的编码值:")
                for i, (pos1, pos2, val1, val2) in enumerate(collisions[:5]):  # 只显示前5个
                    print(f"    Token {pos1} vs Token {pos2}: {val1:.4f} vs {val2:.4f}")
                if len(collisions) > 5:
                    print(f"    ... 还有 {len(collisions) - 5} 对")
            else:
                print(f"  未发现相近的编码值")
            print()
        
        # 可视化高频维度的周期性
        plt.figure(figsize=(15, 12))
        
        # 绘制高频维度的编码值
        for i, dim in enumerate(high_freq_dims):
            plt.subplot(3, 2, i + 1)
            values = pe[:, dim].numpy()
            plt.plot(range(seq_len), values, 'b-', linewidth=2)
            plt.title(f'维度 {dim} 的位置编码值')
            plt.xlabel('Token Position')
            plt.ylabel('Encoding Value')
            plt.grid(True, alpha=0.3)
            
            # 标记相近的点
            tolerance = 0.05
            for pos in range(seq_len):
                similar_positions = []
                for other_pos in range(pos + 1, seq_len):
                    if abs(values[pos] - values[other_pos]) < tolerance:
                        similar_positions.append(other_pos)
                
                if similar_positions:
                    plt.scatter([pos] + similar_positions, 
                              [values[pos]] + [values[p] for p in similar_positions],
                              c='red', s=50, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('high_frequency_collision_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 分析不同频率维度的对比
        print("\n不同频率维度对比:")
        print("="*60)
        
        low_freq_dims = [500, 501, 502, 503, 504, 505]  # 低频维度
        
        print("高频维度 vs 低频维度的值域对比:")
        for i, (high_dim, low_dim) in enumerate(zip(high_freq_dims, low_freq_dims)):
            high_values = pe[:, high_dim].numpy()
            low_values = pe[:, low_dim].numpy()
            
            # 计算值的变化速度
            high_diff = np.diff(high_values)
            low_diff = np.diff(low_values)
            
            print(f"维度 {high_dim} (高频): 平均变化率 = {np.mean(np.abs(high_diff)):.4f}")
            print(f"维度 {low_dim} (低频): 平均变化率 = {np.mean(np.abs(low_diff)):.4f}")
            print(f"变化率比值: {np.mean(np.abs(high_diff)) / np.mean(np.abs(low_diff)):.2f}")
            print()

    def analyze_sinusoidal_benefits(self):
        """分析正弦位置编码的好处"""
        print("\n正弦位置编码的优势分析:")
        print("="*60)
        
        seq_len = 64
        d_model = 128
        pe = self.create_positional_encoding(seq_len, d_model)
        
        # 1. 相对位置表示能力
        print("1. 相对位置表示能力:")
        print("-" * 30)
        
        # 通过内积计算相对位置相关性
        from torch.nn.functional import cosine_similarity
        
        # 测试不同距离的相对位置
        distances = [1, 2, 5, 10, 20]
        for dist in distances:
            similarities = []
            for i in range(seq_len - dist):
                sim = cosine_similarity(pe[i].unsqueeze(0), pe[i + dist].unsqueeze(0))
                similarities.append(sim.item())
            
            avg_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            print(f"  距离 {dist}: 平均相似度 = {avg_sim:.4f} ± {std_sim:.4f}")
        
        # 2. 外推能力测试
        print("\n2. 外推能力测试:")
        print("-" * 30)
        
        # 训练长度 vs 测试长度
        train_len = 32
        test_len = 64
        
        train_pe = self.create_positional_encoding(train_len, d_model)
        test_pe = self.create_positional_encoding(test_len, d_model)
        
        # 比较训练范围内的编码是否一致
        diff = torch.abs(train_pe - test_pe[:train_len])
        print(f"  训练长度内编码一致性: 最大差异 = {diff.max().item():.8f}")
        
        # 测试外推部分的合理性
        extrapolated_pe = test_pe[train_len:]
        print(f"  外推部分形状: {extrapolated_pe.shape}")
        print(f"  外推部分值域: [{extrapolated_pe.min().item():.4f}, {extrapolated_pe.max().item():.4f}]")
        
        # 3. 计算效率
        print("\n3. 计算效率:")
        print("-" * 30)
        
        import time
        
        # 测试生成时间
        start_time = time.time()
        for _ in range(100):
            pe_temp = self.create_positional_encoding(seq_len, d_model)
        end_time = time.time()
        
        print(f"  生成100次位置编码平均时间: {(end_time - start_time) / 100 * 1000:.2f} ms")
        print(f"  无需训练参数，直接计算生成")
        
        # 4. 唯一性验证
        print("\n4. 唯一性验证:")
        print("-" * 30)
        
        # 检查是否有完全相同的编码
        unique_encodings = torch.unique(pe, dim=0)
        print(f"  原始编码数量: {pe.shape[0]}")
        print(f"  唯一编码数量: {unique_encodings.shape[0]}")
        print(f"  编码唯一性: {unique_encodings.shape[0] == pe.shape[0]}")
        
        # 5. 线性变换不变性
        print("\n5. 线性变换特性:")
        print("-" * 30)
        
        # 测试位置编码的线性组合特性
        pos_a, pos_b = 10, 20
        alpha = 0.3
        
        # 理论上：PE(a) + PE(b) 应该与某种线性组合相关
        pe_a = pe[pos_a]
        pe_b = pe[pos_b]
        linear_combo = alpha * pe_a + (1 - alpha) * pe_b
        
        print(f"  位置 {pos_a} 编码范数: {torch.norm(pe_a):.4f}")
        print(f"  位置 {pos_b} 编码范数: {torch.norm(pe_b):.4f}")
        print(f"  线性组合编码范数: {torch.norm(linear_combo):.4f}")

    def create_rope_encoding(self, seq_len, d_model, theta=10000):
        """
        创建RoPE(Rotary Position Embedding)编码
        """
        # 创建频率向量
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # 创建位置索引
        t = torch.arange(seq_len).float()
        
        # 计算角度
        angles = torch.outer(t, freqs)  # [seq_len, d_model//2]
        
        # 创建复数形式的旋转矩阵
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        return cos_angles, sin_angles, freqs

    def apply_rope(self, x, cos_angles, sin_angles):
        """
        应用RoPE到输入张量
        Args:
            x: 输入张量 [seq_len, d_model]
            cos_angles, sin_angles: 旋转角度
        """
        # 将输入重塑为复数对
        x_reshaped = x.view(x.shape[0], -1, 2)  # [seq_len, d_model//2, 2]
        
        # 应用旋转
        x_rotated = torch.zeros_like(x_reshaped)
        x_rotated[:, :, 0] = x_reshaped[:, :, 0] * cos_angles - x_reshaped[:, :, 1] * sin_angles
        x_rotated[:, :, 1] = x_reshaped[:, :, 0] * sin_angles + x_reshaped[:, :, 1] * cos_angles
        
        return x_rotated.view(x.shape)

    def analyze_rope_benefits(self):
        """分析RoPE的优势"""
        print("\n\nRoPE (旋转位置编码) 的优势分析:")
        print("="*60)
        
        seq_len = 64
        d_model = 128
        
        # 生成RoPE编码
        cos_angles, sin_angles, freqs = self.create_rope_encoding(seq_len, d_model)
        
        # 1. 相对位置不变性
        print("1. 相对位置不变性:")
        print("-" * 30)
        
        # 创建两个查询向量
        q1 = torch.randn(d_model)
        q2 = torch.randn(d_model)
        
        # 在不同位置应用RoPE
        positions = [0, 10, 20, 30]
        relative_distances = []
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:
                    # 应用RoPE
                    q1_rotated = self.apply_rope(q1.unsqueeze(0), 
                                               cos_angles[pos1:pos1+1], 
                                               sin_angles[pos1:pos1+1]).squeeze(0)
                    q2_rotated = self.apply_rope(q2.unsqueeze(0), 
                                               cos_angles[pos2:pos2+1], 
                                               sin_angles[pos2:pos2+1]).squeeze(0)
                    
                    # 计算内积
                    dot_product = torch.dot(q1_rotated, q2_rotated)
                    relative_distances.append((pos1, pos2, pos2-pos1, dot_product.item()))
        
        print("  位置对 (pos1, pos2, 距离, 内积):")
        for pos1, pos2, dist, dot in relative_distances:
            print(f"    ({pos1:2d}, {pos2:2d}, {dist:2d}) -> 内积: {dot:8.4f}")
        
        # 2. 频率分析
        print("\n2. 频率分布分析:")
        print("-" * 30)
        
        print(f"  频率范围: [{freqs.min().item():.6f}, {freqs.max().item():.6f}]")
        print(f"  频率数量: {len(freqs)}")
        
        # 显示不同频率对应的周期
        periods = 2 * np.pi / freqs.numpy()
        print(f"  周期范围: [{periods.min():.2f}, {periods.max():.2f}]")
        
        # 3. 外推能力测试
        print("\n3. 外推能力测试:")
        print("-" * 30)
        
        # 训练长度 vs 测试长度
        train_len = 32
        test_len = 128
        
        train_cos, train_sin, _ = self.create_rope_encoding(train_len, d_model)
        test_cos, test_sin, _ = self.create_rope_encoding(test_len, d_model)
        
        # 比较训练范围内的一致性
        cos_diff = torch.abs(train_cos - test_cos[:train_len])
        sin_diff = torch.abs(train_sin - test_sin[:train_len])
        
        print(f"  训练范围内cos一致性: 最大差异 = {cos_diff.max().item():.8f}")
        print(f"  训练范围内sin一致性: 最大差异 = {sin_diff.max().item():.8f}")
        
        # 4. 内积保持性质
        print("\n4. 内积保持性质:")
        print("-" * 30)
        
        # 测试旋转后内积的变化
        x1 = torch.randn(d_model)
        x2 = torch.randn(d_model)
        
        # 原始内积
        original_dot = torch.dot(x1, x2)
        
        # 应用相同旋转后的内积
        pos = 15
        x1_rotated = self.apply_rope(x1.unsqueeze(0), 
                                   cos_angles[pos:pos+1], 
                                   sin_angles[pos:pos+1]).squeeze(0)
        x2_rotated = self.apply_rope(x2.unsqueeze(0), 
                                   cos_angles[pos:pos+1], 
                                   sin_angles[pos:pos+1]).squeeze(0)
        
        rotated_dot = torch.dot(x1_rotated, x2_rotated)
        
        print(f"  原始内积: {original_dot.item():.6f}")
        print(f"  旋转后内积: {rotated_dot.item():.6f}")
        print(f"  内积保持差异: {abs(original_dot.item() - rotated_dot.item()):.8f}")
        
        # 5. 计算效率对比
        print("\n5. 计算效率对比:")
        print("-" * 30)
        
        import time
        
        # RoPE生成时间
        start_time = time.time()
        for _ in range(100):
            cos_temp, sin_temp, _ = self.create_rope_encoding(seq_len, d_model)
        rope_time = time.time() - start_time
        
        # 正弦位置编码生成时间
        start_time = time.time()
        for _ in range(100):
            pe_temp = self.create_positional_encoding(seq_len, d_model)
        sine_time = time.time() - start_time
        
        print(f"  RoPE生成时间: {rope_time/100*1000:.2f} ms")
        print(f"  正弦编码生成时间: {sine_time/100*1000:.2f} ms")
        print(f"  时间比率: {rope_time/sine_time:.2f}x")

    def visualize_rope_vs_sinusoidal(self):
        """可视化RoPE vs 正弦位置编码的对比"""
        seq_len = 64
        d_model = 64
        
        # 生成两种编码
        pe_sine = self.create_positional_encoding(seq_len, d_model)
        cos_angles, sin_angles, freqs = self.create_rope_encoding(seq_len, d_model)
        
        # 可视化对比
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 正弦位置编码热力图
        im1 = axes[0, 0].imshow(pe_sine.numpy(), cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title('Sinusoidal Position Encoding')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Position')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # RoPE cos部分热力图
        im2 = axes[0, 1].imshow(cos_angles.numpy(), cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('RoPE Cosine Angles')
        axes[0, 1].set_xlabel('Frequency Index')
        axes[0, 1].set_ylabel('Position')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # RoPE sin部分热力图
        im3 = axes[0, 2].imshow(sin_angles.numpy(), cmap='RdBu_r', aspect='auto')
        axes[0, 2].set_title('RoPE Sine Angles')
        axes[0, 2].set_xlabel('Frequency Index')
        axes[0, 2].set_ylabel('Position')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 频率分布对比
        sine_freqs = []
        for i in range(0, d_model, 2):
            freq = 1.0 / (10000 ** (i / d_model))
            sine_freqs.append(freq)
        
        axes[1, 0].semilogy(sine_freqs, 'b-o', label='Sinusoidal')
        axes[1, 0].semilogy(freqs.numpy(), 'r-s', label='RoPE')
        axes[1, 0].set_title('Frequency Distribution')
        axes[1, 0].set_xlabel('Dimension/Frequency Index')
        axes[1, 0].set_ylabel('Frequency (log scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 周期分布对比
        sine_periods = [2 * np.pi / f for f in sine_freqs]
        rope_periods = 2 * np.pi / freqs.numpy()
        
        axes[1, 1].semilogy(sine_periods, 'b-o', label='Sinusoidal')
        axes[1, 1].semilogy(rope_periods, 'r-s', label='RoPE')
        axes[1, 1].set_title('Period Distribution')
        axes[1, 1].set_xlabel('Dimension/Frequency Index')
        axes[1, 1].set_ylabel('Period (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 相对位置相关性对比
        distances = range(1, 21)
        sine_correlations = []
        rope_correlations = []
        
        for dist in distances:
            # 正弦编码相关性
            sine_sims = []
            for i in range(seq_len - dist):
                sim = torch.cosine_similarity(pe_sine[i].unsqueeze(0), pe_sine[i + dist].unsqueeze(0))
                sine_sims.append(sim.item())
            sine_correlations.append(np.mean(sine_sims))
            
            # RoPE相关性 (使用角度差来近似)
            rope_angle_diff = torch.mean(torch.abs(cos_angles[dist:] - cos_angles[:-dist]))
            rope_correlations.append(1 - rope_angle_diff.item())
        
        axes[1, 2].plot(distances, sine_correlations, 'b-o', label='Sinusoidal')
        axes[1, 2].plot(distances, rope_correlations, 'r-s', label='RoPE')
        axes[1, 2].set_title('Relative Position Correlation')
        axes[1, 2].set_xlabel('Distance')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rope_vs_sinusoidal_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
