import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal


class GyroAnalyzer:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.file_path = os.path.abspath(excel_file)
        self.df = None
        self.timestamps = None
        self.gyro_x = None
        self.gyro_y = None
        self.gyro_z = None
        self.attitudes = None
        self.figures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures"
        )

        # 确保figures目录存在
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

    def load_data(self):
        """加载并处理Excel数据"""
        print(f"Reading file: {self.file_path}")
        self.df = pd.read_excel(self.file_path)
        self.timestamps = np.array(self.df["Time (s)"].tolist())
        self.gyro_x = np.array(self.df.iloc[:, 1].tolist())
        self.gyro_y = np.array(self.df.iloc[:, 2].tolist())
        self.gyro_z = np.array(self.df.iloc[:, 3].tolist())

    def estimate_attitude(self):
        """使用四元数估计姿态"""
        q = R.identity()
        attitudes = []
        attitudes.append(q.as_euler("xyz", degrees=True))

        for i in range(1, len(self.timestamps)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            omega_avg = 0.5 * (
                np.array([self.gyro_x[i], self.gyro_y[i], self.gyro_z[i]])
                + np.array([self.gyro_x[i - 1], self.gyro_y[i - 1], self.gyro_z[i - 1]])
            )
            angle_vec = omega_avg * dt
            dq = R.from_rotvec(angle_vec)
            q = q * dq
            attitudes.append(q.as_euler("xyz", degrees=True))

        self.attitudes = np.array(attitudes)

    def plot_angular_velocity_and_attitude(self):
        """绘制角速度和姿态图"""
        plt.figure(figsize=(12, 10))

        # 绘制角速度
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.gyro_x, "r-", label="Gyro X")
        plt.plot(self.timestamps, self.gyro_y, "g-", label="Gyro Y")
        plt.plot(self.timestamps, self.gyro_z, "b-", label="Gyro Z")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.title("Angular Velocity Data")
        plt.legend()
        plt.grid(True)

        # 绘制欧拉角
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.attitudes[:, 0], "r-", label="Roll")
        plt.plot(self.timestamps, self.attitudes[:, 1], "g-", label="Pitch")
        plt.plot(self.timestamps, self.attitudes[:, 2], "b-", label="Yaw")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (degrees)")
        plt.title("Attitude (Euler Angles)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        figure_path = os.path.join(self.figures_dir, "gyro_attitude_plot_4_12s.png")
        plt.savefig(figure_path, dpi=300)
        print(f"Figure saved to: {figure_path}")
        plt.show()

    def plot_euler_magnitude_and_spectrum(self):
        """绘制欧拉角幅值和频谱分析图，同时包含角速度模长的频谱"""
        plt.figure(figsize=(15, 12))

        # 计算欧拉角幅值
        euler_magnitude = np.sqrt(np.sum(self.attitudes**2, axis=1))

        # 计算角速度模长
        gyro_magnitude = np.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2)

        # 绘制欧拉角幅值
        plt.subplot(3, 1, 1)
        plt.plot(self.timestamps, euler_magnitude, "k-", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude (degrees)")
        plt.title("Magnitude of Euler Angles")
        plt.grid(True, alpha=0.3)

        # 绘制角速度模长
        plt.subplot(3, 1, 2)
        plt.plot(self.timestamps, gyro_magnitude, "r-", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude (rad/s)")
        plt.title("Magnitude of Angular Velocity")
        plt.grid(True, alpha=0.3)

        # 进行频谱分析
        sample_rate = 1 / np.mean(np.diff(self.timestamps))

        # 欧拉角频谱分析
        # 去除直流分量
        euler_magnitude_ac = euler_magnitude - np.mean(euler_magnitude)
        n_euler = len(euler_magnitude_ac)
        yf_euler = np.fft.rfft(euler_magnitude_ac)
        xf_euler = np.fft.rfftfreq(n_euler, 1 / sample_rate)

        # 角速度频谱分析
        # 去除直流分量
        gyro_magnitude_ac = gyro_magnitude - np.mean(gyro_magnitude)
        n_gyro = len(gyro_magnitude_ac)
        yf_gyro = np.fft.rfft(gyro_magnitude_ac)
        xf_gyro = np.fft.rfftfreq(n_gyro, 1 / sample_rate)

        # 绘制频谱对比
        plt.subplot(3, 1, 3)
        # 归一化频谱幅值 (除以数据长度)
        plt.plot(
            xf_euler,
            np.abs(yf_euler) / n_euler,
            "b-",
            label="Euler Angle Magnitude",
            linewidth=1.5,
        )
        plt.plot(
            xf_gyro,
            np.abs(yf_gyro) / n_gyro,
            "r-",
            label="Angular Velocity Magnitude",
            linewidth=1.5,
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Amplitude")
        plt.title("Frequency Spectrum Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 限制频率范围以便更好地观察
        plt.xlim(0, min(50, sample_rate / 2))  # 显示0-50Hz或奈奎斯特频率的一半

        plt.tight_layout()

        freq_figure_path = os.path.join(
            self.figures_dir, "euler_gyro_magnitude_spectrum_comparison.png"
        )
        plt.savefig(freq_figure_path, dpi=300, bbox_inches="tight")
        print(f"Frequency analysis comparison saved to: {freq_figure_path}")
        plt.show()

        # 打印一些统计信息
        print(f"采样频率: {sample_rate:.2f} Hz")
        print(
            f"欧拉角幅值范围: {np.min(euler_magnitude):.2f} - {np.max(euler_magnitude):.2f} degrees"
        )
        print(
            f"角速度幅值范围: {np.min(gyro_magnitude):.4f} - {np.max(gyro_magnitude):.4f} rad/s"
        )
        print(
            f"欧拉角主要频率成分: {xf_euler[np.argmax(np.abs(yf_euler)[1:]) + 1]:.2f} Hz"
        )
        print(
            f"角速度主要频率成分: {xf_gyro[np.argmax(np.abs(yf_gyro)[1:]) + 1]:.2f} Hz"
        )

    def plot_power_spectral_density(self):
        """绘制功率谱密度对比"""
        plt.figure(figsize=(15, 8))

        # 计算欧拉角幅值和角速度模长
        euler_magnitude = np.sqrt(np.sum(self.attitudes**2, axis=1))
        gyro_magnitude = np.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2)

        # 计算采样频率
        sample_rate = 1 / np.mean(np.diff(self.timestamps))

        # 使用Welch方法计算功率谱密度
        # 去除直流分量
        euler_magnitude_ac = euler_magnitude - np.mean(euler_magnitude)
        gyro_magnitude_ac = gyro_magnitude - np.mean(gyro_magnitude)

        # 计算PSD
        f_euler, psd_euler = signal.welch(
            euler_magnitude_ac,
            sample_rate,
            nperseg=min(256, len(euler_magnitude_ac) // 4),
        )
        f_gyro, psd_gyro = signal.welch(
            gyro_magnitude_ac,
            sample_rate,
            nperseg=min(256, len(gyro_magnitude_ac) // 4),
        )

        # 绘制PSD对比
        plt.subplot(1, 2, 1)
        plt.semilogy(
            f_euler, psd_euler, "b-", label="Euler Angle Magnitude", linewidth=1.5
        )
        plt.semilogy(
            f_gyro, psd_gyro, "r-", label="Angular Velocity Magnitude", linewidth=1.5
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.title("Power Spectral Density Comparison (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(25, sample_rate / 2))

        # 线性尺度的PSD
        plt.subplot(1, 2, 2)
        plt.plot(f_euler, psd_euler, "b-", label="Euler Angle Magnitude", linewidth=1.5)
        plt.plot(
            f_gyro, psd_gyro, "r-", label="Angular Velocity Magnitude", linewidth=1.5
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.title("Power Spectral Density Comparison (Linear Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(25, sample_rate / 2))

        plt.tight_layout()

        psd_figure_path = os.path.join(
            self.figures_dir, "power_spectral_density_comparison.png"
        )
        plt.savefig(psd_figure_path, dpi=300, bbox_inches="tight")
        print(f"Power spectral density comparison saved to: {psd_figure_path}")
        plt.show()

        # 找到主要频率成分
        main_freq_euler = f_euler[np.argmax(psd_euler)]
        main_freq_gyro = f_gyro[np.argmax(psd_gyro)]
        print(f"欧拉角主要频率成分 (PSD): {main_freq_euler:.2f} Hz")
        print(f"角速度主要频率成分 (PSD): {main_freq_gyro:.2f} Hz")

    def compare_spectrum_methods(self):
        """详细比较FFT和Welch方法的差异"""
        print("\n" + "=" * 60)
        print("两种频谱分析方法的差异解释：")
        print("=" * 60)

        # 计算数据
        euler_magnitude = np.sqrt(np.sum(self.attitudes**2, axis=1))
        gyro_magnitude = np.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2)
        sample_rate = 1 / np.mean(np.diff(self.timestamps))

        # 去除直流分量
        euler_magnitude_ac = euler_magnitude - np.mean(euler_magnitude)
        gyro_magnitude_ac = gyro_magnitude - np.mean(gyro_magnitude)

        # FFT分析
        yf_euler = np.fft.rfft(euler_magnitude_ac)
        xf_euler = np.fft.rfftfreq(len(euler_magnitude_ac), 1 / sample_rate)
        yf_gyro = np.fft.rfft(gyro_magnitude_ac)
        xf_gyro = np.fft.rfftfreq(len(gyro_magnitude_ac), 1 / sample_rate)

        # Welch方法
        nperseg = min(256, len(euler_magnitude_ac) // 4)
        f_euler_welch, psd_euler = signal.welch(
            euler_magnitude_ac, sample_rate, nperseg=nperseg
        )
        f_gyro_welch, psd_gyro = signal.welch(
            gyro_magnitude_ac, sample_rate, nperseg=nperseg
        )

        # 找到主要频率
        # FFT方法 (排除0Hz)
        fft_euler_peak_idx = np.argmax(np.abs(yf_euler)[1:]) + 1
        fft_gyro_peak_idx = np.argmax(np.abs(yf_gyro)[1:]) + 1
        fft_euler_freq = xf_euler[fft_euler_peak_idx]
        fft_gyro_freq = xf_gyro[fft_gyro_peak_idx]

        # Welch方法
        welch_euler_freq = f_euler_welch[np.argmax(psd_euler)]
        welch_gyro_freq = f_gyro_welch[np.argmax(psd_gyro)]

        print(f"1. 方法差异：")
        print(f"   - FFT方法：使用整个数据序列进行快速傅里叶变换")
        print(f"   - Welch方法：将数据分段，对每段做FFT后取平均，降低噪声影响")
        print(f"   - Welch分段长度：{nperseg} 个采样点")
        print(f"   - 数据总长度：{len(euler_magnitude_ac)} 个采样点")

        print(f"\n2. 频率分辨率差异：")
        freq_res_fft = sample_rate / len(euler_magnitude_ac)
        freq_res_welch = sample_rate / nperseg
        print(f"   - FFT频率分辨率：{freq_res_fft:.4f} Hz")
        print(f"   - Welch频率分辨率：{freq_res_welch:.4f} Hz")

        print(f"\n3. 主要频率成分对比：")
        print(f"   欧拉角幅值：")
        print(f"     - FFT方法：{fft_euler_freq:.4f} Hz")
        print(f"     - Welch方法：{welch_euler_freq:.4f} Hz")
        print(f"   角速度幅值：")
        print(f"     - FFT方法：{fft_gyro_freq:.4f} Hz")
        print(f"     - Welch方法：{welch_gyro_freq:.4f} Hz")

        print(f"\n4. 差异原因：")
        print(f"   - FFT对噪声敏感，可能检测到噪声峰值")
        print(f"   - Welch方法通过平均减少噪声，更稳定")
        print(f"   - 不同频率分辨率导致峰值位置略有不同")
        print(f"   - 信号可能有多个接近的频率成分")

        # 创建详细对比图
        plt.figure(figsize=(15, 10))

        # 时域信号
        plt.subplot(2, 2, 1)
        plt.plot(
            self.timestamps,
            euler_magnitude_ac,
            "b-",
            alpha=0.7,
            label="Euler Magnitude (AC)",
        )
        plt.plot(
            self.timestamps,
            gyro_magnitude_ac,
            "r-",
            alpha=0.7,
            label="Gyro Magnitude (AC)",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Time Domain Signals (DC Removed)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # FFT结果
        plt.subplot(2, 2, 2)
        plt.plot(
            xf_euler[: len(xf_euler) // 10],
            np.abs(yf_euler)[: len(yf_euler) // 10] / len(euler_magnitude_ac),
            "b-",
            label="Euler (FFT)",
        )
        plt.plot(
            xf_gyro[: len(xf_gyro) // 10],
            np.abs(yf_gyro)[: len(yf_gyro) // 10] / len(gyro_magnitude_ac),
            "r-",
            label="Gyro (FFT)",
        )
        plt.axvline(
            fft_euler_freq,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Euler Peak: {fft_euler_freq:.3f}Hz",
        )
        plt.axvline(
            fft_gyro_freq,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Gyro Peak: {fft_gyro_freq:.3f}Hz",
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Magnitude")
        plt.title("FFT Spectrum")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Welch结果
        plt.subplot(2, 2, 3)
        plt.plot(f_euler_welch, psd_euler, "b-", label="Euler (Welch)")
        plt.plot(f_gyro_welch, psd_gyro, "r-", label="Gyro (Welch)")
        plt.axvline(
            welch_euler_freq,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Euler Peak: {welch_euler_freq:.3f}Hz",
        )
        plt.axvline(
            welch_gyro_freq,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Gyro Peak: {welch_gyro_freq:.3f}Hz",
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.title("Welch Power Spectral Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 5)  # 聚焦低频部分

        # 对数尺度对比
        plt.subplot(2, 2, 4)
        plt.semilogy(f_euler_welch, psd_euler, "b-", label="Euler (Welch)")
        plt.semilogy(f_gyro_welch, psd_gyro, "r-", label="Gyro (Welch)")
        plt.axvline(welch_euler_freq, color="blue", linestyle="--", alpha=0.7)
        plt.axvline(welch_gyro_freq, color="red", linestyle="--", alpha=0.7)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (log)")
        plt.title("Welch PSD (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 5)

        plt.tight_layout()

        comparison_path = os.path.join(
            self.figures_dir, "spectrum_methods_comparison.png"
        )
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        print(f"\n详细对比图保存至: {comparison_path}")
        plt.show()

    def analyze(self):
        """执行完整的分析流程"""
        self.load_data()
        self.estimate_attitude()
        # self.plot_angular_velocity_and_attitude()
        self.plot_euler_magnitude_and_spectrum()
        self.plot_power_spectral_density()
        self.compare_spectrum_methods()


def main():
    excel_file = os.path.join("data/iphone_imu", "velocitytoangle.xls")
    analyzer = GyroAnalyzer(excel_file)
    analyzer.analyze()


if __name__ == "__main__":
    main()
