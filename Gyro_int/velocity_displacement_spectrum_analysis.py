import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class VelocityDisplacementAnalyzer:
    def __init__(self, fs=1000, duration=10):
        """
        初始化分析器
        
        Parameters:
        fs: 采样频率 (Hz)
        duration: 信号持续时间 (秒)
        """
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        self.dt = 1 / fs
        
    def generate_velocity_signals(self):
        """生成不同类型的速度信号"""
        signals = {}
        
        # 1. 单一正弦波速度
        f1 = 2  # Hz
        signals['sine_2Hz'] = {
            'velocity': np.sin(2 * np.pi * f1 * self.t),
            'description': '单一正弦波 (2Hz)'
        }
        
        # 2. 复合正弦波速度
        f2, f3 = 5, 10  # Hz
        signals['composite_sine'] = {
            'velocity': np.sin(2 * np.pi * f2 * self.t) + 0.5 * np.sin(2 * np.pi * f3 * self.t),
            'description': '复合正弦波 (5Hz + 10Hz)'
        }
        
        # 3. 带噪声的正弦波速度
        noise_level = 0.1
        signals['noisy_sine'] = {
            'velocity': np.sin(2 * np.pi * f1 * self.t) + noise_level * np.random.randn(len(self.t)),
            'description': '带噪声的正弦波 (2Hz + 噪声)'
        }
        
        # 4. 线性调频信号（chirp）
        f_start, f_end = 1, 20
        signals['chirp'] = {
            'velocity': signal.chirp(self.t, f_start, self.duration, f_end),
            'description': f'线性调频信号 ({f_start}-{f_end}Hz)'
        }
        
        return signals
    
    def velocity_to_displacement(self, velocity):
        """
        通过数值积分将速度转换为位移
        
        Parameters:
        velocity: 速度数组
        
        Returns:
        displacement: 位移数组
        """
        # 使用累积梯形积分
        displacement = cumtrapz(velocity, dx=self.dt, initial=0)
        return displacement
    
    def compute_spectrum(self, signal_data):
        """
        计算信号的频谱
        
        Parameters:
        signal_data: 输入信号
        
        Returns:
        freqs: 频率数组
        magnitude: 幅度谱
        phase: 相位谱
        """
        # 计算FFT
        fft_result = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), self.dt)
        
        # 只保留正频率部分
        positive_freq_idx = freqs >= 0
        freqs = freqs[positive_freq_idx]
        fft_result = fft_result[positive_freq_idx]
        
        # 计算幅度谱和相位谱
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        return freqs, magnitude, phase
    
    def analyze_velocity_displacement_relationship(self):
        """分析速度和位移的频谱关系"""
        # 生成速度信号
        velocity_signals = self.generate_velocity_signals()
        
        # 创建图形
        fig, axes = plt.subplots(len(velocity_signals), 4, figsize=(20, 5*len(velocity_signals)))
        if len(velocity_signals) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (signal_name, signal_info) in enumerate(velocity_signals.items()):
            velocity = signal_info['velocity']
            displacement = self.velocity_to_displacement(velocity)
            
            # 计算频谱
            v_freqs, v_magnitude, v_phase = self.compute_spectrum(velocity)
            d_freqs, d_magnitude, d_phase = self.compute_spectrum(displacement)
            
            # 绘制时域信号
            axes[i, 0].plot(self.t[:1000], velocity[:1000], 'b-', label='速度')
            axes[i, 0].set_title(f'{signal_info["description"]} - 时域')
            axes[i, 0].set_xlabel('时间 (s)')
            axes[i, 0].set_ylabel('速度 (m/s)')
            axes[i, 0].grid(True)
            axes[i, 0].legend()
            
            axes[i, 1].plot(self.t[:1000], displacement[:1000], 'r-', label='位移')
            axes[i, 1].set_title(f'{signal_info["description"]} - 位移时域')
            axes[i, 1].set_xlabel('时间 (s)')
            axes[i, 1].set_ylabel('位移 (m)')
            axes[i, 1].grid(True)
            axes[i, 1].legend()
            
            # 绘制幅度谱
            freq_limit = 25  # 只显示25Hz以下的频率
            freq_mask = v_freqs <= freq_limit
            
            axes[i, 2].semilogy(v_freqs[freq_mask], v_magnitude[freq_mask], 'b-', label='速度幅度谱')
            axes[i, 2].semilogy(d_freqs[freq_mask], d_magnitude[freq_mask], 'r-', label='位移幅度谱')
            axes[i, 2].set_title(f'{signal_info["description"]} - 幅度谱')
            axes[i, 2].set_xlabel('频率 (Hz)')
            axes[i, 2].set_ylabel('幅度')
            axes[i, 2].grid(True)
            axes[i, 2].legend()
            
            # 绘制相位谱
            axes[i, 3].plot(v_freqs[freq_mask], v_phase[freq_mask], 'b-', label='速度相位谱')
            axes[i, 3].plot(d_freqs[freq_mask], d_phase[freq_mask], 'r-', label='位移相位谱')
            axes[i, 3].set_title(f'{signal_info["description"]} - 相位谱')
            axes[i, 3].set_xlabel('频率 (Hz)')
            axes[i, 3].set_ylabel('相位 (rad)')
            axes[i, 3].grid(True)
            axes[i, 3].legend()
        
        plt.tight_layout()
        plt.savefig('figures/velocity_displacement_spectrum_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_sine_wave_fft(self):
        """分析正弦波的FFT变换特性"""
        print("=" * 60)
        print("正弦波FFT变换分析")
        print("=" * 60)
        
        # 生成标准正弦波
        freq = 5  # Hz
        amplitude = 2
        phase = np.pi/4  # 初始相位
        
        sine_wave = amplitude * np.sin(2 * np.pi * freq * self.t + phase)
        
        # 计算FFT
        freqs, magnitude, phase_spectrum = self.compute_spectrum(sine_wave)
        
        # 找到主频率分量
        main_freq_idx = np.argmax(magnitude)
        detected_freq = freqs[main_freq_idx]
        detected_amplitude = magnitude[main_freq_idx] * 2 / len(self.t)  # 校正幅度
        detected_phase = phase_spectrum[main_freq_idx]
        
        print(f"原始正弦波参数:")
        print(f"  频率: {freq} Hz")
        print(f"  幅度: {amplitude}")
        print(f"  相位: {phase:.4f} rad ({np.degrees(phase):.2f}°)")
        print()
        print(f"FFT检测结果:")
        print(f"  检测频率: {detected_freq:.4f} Hz")
        print(f"  检测幅度: {detected_amplitude:.4f}")
        print(f"  检测相位: {detected_phase:.4f} rad ({np.degrees(detected_phase):.2f}°)")
        print()
        
        # 验证FFT的可逆性
        reconstructed = np.fft.ifft(np.fft.fft(sine_wave)).real
        reconstruction_error = np.mean(np.abs(sine_wave - reconstructed))
        print(f"重构误差: {reconstruction_error:.2e}")
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 时域信号
        axes[0, 0].plot(self.t[:200], sine_wave[:200], 'b-', linewidth=2)
        axes[0, 0].set_title(f'原始正弦波 (f={freq}Hz, A={amplitude})')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].grid(True)
        
        # 重构信号
        axes[0, 1].plot(self.t[:200], sine_wave[:200], 'b-', linewidth=2, label='原始')
        axes[0, 1].plot(self.t[:200], reconstructed[:200], 'r--', linewidth=2, label='重构')
        axes[0, 1].set_title('原始 vs 重构信号')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 幅度谱
        freq_mask = (freqs >= 0) & (freqs <= 20)
        axes[1, 0].stem(freqs[freq_mask], magnitude[freq_mask], basefmt=' ')
        axes[1, 0].set_title('幅度谱')
        axes[1, 0].set_xlabel('频率 (Hz)')
        axes[1, 0].set_ylabel('幅度')
        axes[1, 0].grid(True)
        
        # 相位谱
        axes[1, 1].stem(freqs[freq_mask], phase_spectrum[freq_mask], basefmt=' ')
        axes[1, 1].set_title('相位谱')
        axes[1, 1].set_xlabel('频率 (Hz)')
        axes[1, 1].set_ylabel('相位 (rad)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('figures/sine_wave_fft_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def theoretical_analysis(self):
        """理论分析速度和位移的频谱关系"""
        print("=" * 60)
        print("速度和位移频谱关系的理论分析")
        print("=" * 60)
        print()
        print("在频域中，积分对应于除以 jω（其中 j 是虚数单位，ω 是角频率）")
        print()
        print("如果速度的频谱为 V(ω)，那么位移的频谱为：")
        print("X(ω) = V(ω) / (jω)")
        print()
        print("这意味着：")
        print("1. 幅度关系：|X(ω)| = |V(ω)| / ω")
        print("2. 相位关系：∠X(ω) = ∠V(ω) - π/2")
        print()
        print("对于正弦函数 v(t) = A*sin(2πft + φ)：")
        print("- 速度频谱在频率 f 处有峰值，幅度为 A，相位为 φ - π/2")
        print("- 位移频谱在频率 f 处有峰值，幅度为 A/(2πf)，相位为 φ - π")
        print()
        
        # 验证理论关系
        f = 5  # Hz
        A = 1  # 幅度
        omega = 2 * np.pi * f
        
        velocity = A * np.sin(omega * self.t)
        displacement_analytical = -A / omega * np.cos(omega * self.t)  # 解析解
        displacement_numerical = self.velocity_to_displacement(velocity)
        
        # 比较解析解和数值解
        error = np.mean(np.abs(displacement_analytical - displacement_numerical))
        print(f"数值积分误差（与解析解比较）: {error:.6f}")

def main():
    """主函数"""
    # 确保figures目录存在
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # 创建分析器
    analyzer = VelocityDisplacementAnalyzer(fs=1000, duration=5)
    
    # 理论分析
    analyzer.theoretical_analysis()
    
    # 分析正弦波FFT特性
    analyzer.analyze_sine_wave_fft()
    
    # 分析速度和位移的频谱关系
    analyzer.analyze_velocity_displacement_relationship()
    
    print("\n分析完成！图像已保存到 figures/ 目录中。")

if __name__ == "__main__":
    main() 