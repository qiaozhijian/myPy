import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz
import os

class VelocityDisplacementAnalyzer:
    def __init__(self, fs=1000, duration=5):
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
        
    def velocity_to_displacement(self, velocity):
        """通过数值积分将速度转换为位移"""
        displacement = cumtrapz(velocity, dx=self.dt, initial=0)
        return displacement
    
    def compute_spectrum(self, signal_data):
        """计算信号的频谱"""
        fft_result = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), self.dt)
        
        # 只保留正频率部分
        positive_freq_idx = freqs >= 0
        freqs = freqs[positive_freq_idx]
        fft_result = fft_result[positive_freq_idx]
        
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        return freqs, magnitude, phase
    
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
        detected_amplitude = magnitude[main_freq_idx] * 2 / len(self.t)
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
        print()
        
        return freq, amplitude, phase, detected_freq, detected_amplitude, detected_phase
    
    def analyze_velocity_displacement_relationship(self):
        """分析速度和位移的频谱关系"""
        print("=" * 60)
        print("速度和位移频谱关系分析")
        print("=" * 60)
        
        # 生成单一正弦波速度
        freq = 2  # Hz
        velocity = np.sin(2 * np.pi * freq * self.t)
        displacement = self.velocity_to_displacement(velocity)
        
        # 计算频谱
        v_freqs, v_magnitude, v_phase = self.compute_spectrum(velocity)
        d_freqs, d_magnitude, d_phase = self.compute_spectrum(displacement)
        
        # 找到主频率分量
        v_main_idx = np.argmax(v_magnitude)
        d_main_idx = np.argmax(d_magnitude)
        
        v_freq = v_freqs[v_main_idx]
        v_amp = v_magnitude[v_main_idx] * 2 / len(self.t)
        v_ph = v_phase[v_main_idx]
        
        d_freq = d_freqs[d_main_idx]
        d_amp = d_magnitude[d_main_idx] * 2 / len(self.t)
        d_ph = d_phase[d_main_idx]
        
        print(f"速度信号 (频率 {freq} Hz):")
        print(f"  检测频率: {v_freq:.4f} Hz")
        print(f"  检测幅度: {v_amp:.4f}")
        print(f"  检测相位: {v_ph:.4f} rad ({np.degrees(v_ph):.2f}°)")
        print()
        print(f"位移信号:")
        print(f"  检测频率: {d_freq:.4f} Hz")
        print(f"  检测幅度: {d_amp:.4f}")
        print(f"  检测相位: {d_ph:.4f} rad ({np.degrees(d_ph):.2f}°)")
        print()
        
        # 理论计算
        omega = 2 * np.pi * freq
        theoretical_d_amp = v_amp / omega
        theoretical_phase_shift = np.pi / 2  # 90度相位滞后
        
        print(f"理论分析:")
        print(f"  理论位移幅度: {theoretical_d_amp:.4f}")
        print(f"  实际位移幅度: {d_amp:.4f}")
        print(f"  幅度比 (位移/速度): {d_amp/v_amp:.4f} (理论值: {1/omega:.4f})")
        print(f"  相位差: {abs(d_ph - v_ph):.4f} rad (理论值: {theoretical_phase_shift:.4f} rad)")
        print()
        
        # 创建简单的可视化
        self.create_visualization(velocity, displacement, v_freqs, v_magnitude, d_freqs, d_magnitude)
        
    def create_visualization(self, velocity, displacement, v_freqs, v_magnitude, d_freqs, d_magnitude):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 时域信号
        time_samples = 1000
        axes[0, 0].plot(self.t[:time_samples], velocity[:time_samples], 'b-', label='Velocity')
        axes[0, 0].set_title('Velocity Signal (Time Domain)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        axes[0, 1].plot(self.t[:time_samples], displacement[:time_samples], 'r-', label='Displacement')
        axes[0, 1].set_title('Displacement Signal (Time Domain)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Displacement (m)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 频域信号
        freq_limit = 10
        freq_mask_v = v_freqs <= freq_limit
        freq_mask_d = d_freqs <= freq_limit
        
        axes[1, 0].semilogy(v_freqs[freq_mask_v], v_magnitude[freq_mask_v], 'b-', label='Velocity Spectrum')
        axes[1, 0].set_title('Velocity Magnitude Spectrum')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        axes[1, 1].semilogy(d_freqs[freq_mask_d], d_magnitude[freq_mask_d], 'r-', label='Displacement Spectrum')
        axes[1, 1].set_title('Displacement Magnitude Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('figures/velocity_displacement_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存到 figures/velocity_displacement_analysis.png")
    
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

def main():
    """主函数"""
    # 确保figures目录存在
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
    
    print("=" * 60)
    print("关于你的问题：如果速度v=sin，FFT变换也会是sin吗？")
    print("=" * 60)
    print("答案：不完全是。FFT变换的结果是复数，包含幅度和相位信息。")
    print("对于正弦函数v(t) = sin(2πft)：")
    print("1. FFT结果在频率f处会有一个峰值（冲激函数）")
    print("2. 频率确实保持不变")
    print("3. 幅值信息保存在FFT结果的幅度中")
    print("4. 相位信息也会被保留")
    print("5. FFT是可逆的，可以完美重构原始信号")
    print()
    print("分析完成！")

if __name__ == "__main__":
    main() 