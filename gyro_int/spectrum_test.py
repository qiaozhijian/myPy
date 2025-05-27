"""
频谱分析测试模块 - 简化版本

主要功能：
1. 快速FFT频谱分析测试
2. 美观的图形显示
3. 英文标签避免字体问题

使用方法：
python gyro_int/spectrum_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


class SpectrumTest:
    def __init__(self):
        self.figures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures"
        )
        # 确保figures目录存在
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

    def analyze_spectrum_fft_only(self, t, signal_data, sample_rate):
        """只进行FFT频谱分析"""

        # FFT分析
        n = len(signal_data)
        yf = np.fft.rfft(signal_data)
        xf = np.fft.rfftfreq(n, 1 / sample_rate)

        # 去除直流分量后的FFT分析
        signal_ac = signal_data - np.mean(signal_data)
        yf_ac = np.fft.rfft(signal_ac)

        # 找到主要频率成分（排除直流）
        fft_peaks_idx = []
        fft_peaks_freq = []
        fft_peaks_amp = []

        # 找前3个最大的峰值（排除DC）
        magnitude_spectrum = np.abs(yf_ac)[1:]  # 排除0Hz
        freq_spectrum = xf[1:]

        for i in range(3):
            if len(magnitude_spectrum) > 0:
                peak_idx = np.argmax(magnitude_spectrum)
                if (
                    magnitude_spectrum[peak_idx] > np.max(magnitude_spectrum) * 0.1
                ):  # 只考虑显著峰值
                    fft_peaks_idx.append(peak_idx)
                    fft_peaks_freq.append(freq_spectrum[peak_idx])
                    fft_peaks_amp.append(
                        magnitude_spectrum[peak_idx] / n * 2
                    )  # 转换为实际幅度

                    # 移除这个峰值周围的点，避免重复检测
                    start_idx = max(0, peak_idx - 2)
                    end_idx = min(len(magnitude_spectrum), peak_idx + 3)
                    magnitude_spectrum[start_idx:end_idx] = 0
                else:
                    break

        return {
            "fft_freq": xf,
            "fft_magnitude": np.abs(yf),
            "fft_magnitude_ac": np.abs(yf_ac),
            "fft_peaks_freq": fft_peaks_freq,
            "fft_peaks_amp": fft_peaks_amp,
        }

    def plot_simple_analysis(
        self, t, signal_clean, signal_noisy, sample_rate, true_freqs, true_amps
    ):
        """简化的FFT频谱分析图"""

        # 设置现代样式
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                plt.style.use("default")

        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

        # 创建1x2布局
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("FFT Spectrum Analysis", fontsize=16, fontweight="bold", y=0.95)

        # 定义颜色方案
        colors = ["#1f77b4", "#ff7f0e"]  # 蓝色、橙色

        # 1. 时域信号对比
        ax1 = axes[0]
        ax1.plot(
            t[:300],
            signal_clean[:300],
            color=colors[0],
            linewidth=2,
            label="Clean Signal",
            alpha=0.8,
        )
        ax1.plot(
            t[:300],
            signal_noisy[:300],
            color=colors[1],
            linewidth=1.5,
            label="Noisy Signal",
            alpha=0.7,
        )
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylabel("Amplitude", fontsize=12)
        ax1.set_title(
            "Time Domain Signals (First 3s)", fontsize=14, fontweight="bold", pad=15
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # 2. FFT频谱对比
        ax2 = axes[1]

        # 纯净信号FFT
        result_clean = self.analyze_spectrum_fft_only(t, signal_clean, sample_rate)
        ax2.plot(
            result_clean["fft_freq"],
            result_clean["fft_magnitude"] / len(signal_clean),
            color=colors[0],
            linewidth=2.5,
            alpha=0.8,
            label="Clean Signal",
        )

        # 加噪声信号FFT
        result_noisy = self.analyze_spectrum_fft_only(t, signal_noisy, sample_rate)
        ax2.plot(
            result_noisy["fft_freq"],
            result_noisy["fft_magnitude"] / len(signal_noisy),
            color=colors[1],
            linewidth=2,
            alpha=0.7,
            label="Noisy Signal",
        )

        # 标记真实频率
        for freq in true_freqs:
            ax2.axvline(freq, color="red", linestyle="--", alpha=0.8, linewidth=2)
            ax2.text(
                freq,
                ax2.get_ylim()[1] * 0.9,
                f"{freq}Hz",
                ha="center",
                fontsize=10,
                color="red",
                fontweight="bold",
            )

        ax2.set_xlabel("Frequency (Hz)", fontsize=12)
        ax2.set_ylabel("Normalized Magnitude", fontsize=12)
        ax2.set_title("FFT Spectrum Comparison", fontsize=14, fontweight="bold", pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 15)
        ax2.legend()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

        # 保存图像
        simple_figure_path = os.path.join(self.figures_dir, "spectrum_test_simple.png")
        plt.savefig(
            simple_figure_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"\nSimple FFT analysis saved to: {simple_figure_path}")
        plt.show()


def quick_test():
    """快速FFT频谱分析测试"""
    test = SpectrumTest()

    print("Quick FFT Spectrum Analysis Test...")
    print("Generating test signals...")

    # 生成简单的测试信号
    duration = 5.0
    sample_rate = 100.0
    t = np.arange(0, duration, 1 / sample_rate)

    # 简单的3频率信号
    freq1, freq2, freq3 = 2.0, 5.0, 8.0
    amp1, amp2, amp3 = 3.0, 2.0, 1.0

    signal_clean = (
        amp1 * np.sin(2 * np.pi * freq1 * t)
        + amp2 * np.sin(2 * np.pi * freq2 * t)
        + amp3 * np.sin(2 * np.pi * freq3 * t)
    )

    noise = 0.5 * np.random.randn(len(t))
    signal_noisy = signal_clean + noise

    true_freqs = [freq1, freq2, freq3]
    true_amps = [amp1, amp2, amp3]

    # 分析和显示结果
    result_clean = test.analyze_spectrum_fft_only(t, signal_clean, sample_rate)
    result_noisy = test.analyze_spectrum_fft_only(t, signal_noisy, sample_rate)

    print(f"\nSignal parameters:")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Duration: {duration} s")
    print(f"  - True frequencies: {true_freqs} Hz")
    print(f"  - True amplitudes: {true_amps}")

    print(f"\nClean signal detected peaks:")
    for freq, amp in zip(result_clean["fft_peaks_freq"], result_clean["fft_peaks_amp"]):
        print(f"  - {freq:.2f} Hz (amplitude: {amp:.2f})")

    print(f"\nNoisy signal detected peaks:")
    for freq, amp in zip(result_noisy["fft_peaks_freq"], result_noisy["fft_peaks_amp"]):
        print(f"  - {freq:.2f} Hz (amplitude: {amp:.2f})")

    # 绘制图形
    test.plot_simple_analysis(
        t, signal_clean, signal_noisy, sample_rate, true_freqs, true_amps
    )

    print("\nTest completed!")


if __name__ == "__main__":
    quick_test()
