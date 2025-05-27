import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    # Get the Excel file path
    excel_file = os.path.join("data", "velocitytoangle.xls")
    file_path = os.path.abspath(excel_file)

    print(f"Reading file: {file_path}")

    # Read the Excel file and analyze the content
    df = pd.read_excel(file_path)
    column_names = df.columns.tolist()
    timestamps = np.array(df["Time (s)"].tolist())  # unit: seconds
    gyro_x = np.array(df.iloc[:, 1].tolist())  # unit: rad/s
    gyro_y = np.array(df.iloc[:, 2].tolist())  # unit: rad/s
    gyro_z = np.array(df.iloc[:, 3].tolist())  # unit: rad/s

    # Filter data to only include 4-12 seconds range
    # mask = (timestamps >= 4) & (timestamps <= 12)
    # timestamps = timestamps[mask]
    # gyro_x = gyro_x[mask]
    # gyro_y = gyro_y[mask]
    # gyro_z = gyro_z[mask]
    
    print(f"Filtered data to time range 4-12 seconds, {len(timestamps)} samples remaining")

    # Attitude estimation using quaternions, and then plot the angular velocity and attitude (Euler angles)
    # Initialize quaternion (unit quaternion represents initial attitude)
    q = R.identity()

    # Store attitude history
    attitudes = []
    attitudes.append(q.as_euler('xyz', degrees=True))  # Convert to Euler angles and store

    # Estimate attitude based on angular velocity data integration
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]  # Calculate time interval
        
        # Integral using the trapezoidal rule (average of the current and previous values)
        omega_avg = 0.5 * (
            np.array([gyro_x[i], gyro_y[i], gyro_z[i]]) + 
            np.array([gyro_x[i-1], gyro_y[i-1], gyro_z[i-1]])
        )
        
        # Angular velocity multiplied by time gives the angular increment
        angle_vec = omega_avg * dt
        
        # Create a rotation representing this increment
        dq = R.from_rotvec(angle_vec)
        
        # Update attitude quaternion
        q = q * dq
        
        # Convert to Euler angles and store
        attitudes.append(q.as_euler('xyz', degrees=True))

    # Convert attitude history to numpy array for plotting
    attitudes = np.array(attitudes)

    # Plot angular velocity and attitude
    plt.figure(figsize=(12, 10))

    # Plot angular velocity
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, gyro_x, 'r-', label='Gyro X')
    plt.plot(timestamps, gyro_y, 'g-', label='Gyro Y')
    plt.plot(timestamps, gyro_z, 'b-', label='Gyro Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity Data')
    plt.legend()
    plt.grid(True)

    # Plot Euler angles
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, attitudes[:, 0], 'r-', label='Roll')
    plt.plot(timestamps, attitudes[:, 1], 'g-', label='Pitch')
    plt.plot(timestamps, attitudes[:, 2], 'b-', label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Attitude (Euler Angles)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Ensure figures folder exists
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Save the plot to figures folder
    figure_path = os.path.join(figures_dir, 'gyro_attitude_plot_4_12s.png')
    plt.savefig(figure_path, dpi=300)
    print(f"Figure saved to: {figure_path}")
    
    # Show the plot
    plt.show()
    
    # Create a new figure for Euler angle magnitude and frequency analysis
    plt.figure(figsize=(12, 10))
    
    # Calculate magnitude of Euler angles (L2 norm)
    euler_magnitude = np.sqrt(np.sum(attitudes**2, axis=1))
    
    # Plot magnitude of Euler angles over time
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, euler_magnitude, 'k-')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude (degrees)')
    plt.title('Magnitude of Euler Angles (4-12s)')
    plt.grid(True)
    
    # Perform frequency analysis using FFT
    sample_rate = 1 / np.mean(np.diff(timestamps))  # Calculate average sample rate
    n = len(euler_magnitude)
    
    # Compute FFT
    yf = np.fft.rfft(euler_magnitude)
    xf = np.fft.rfftfreq(n, 1/sample_rate)
    
    # Plot frequency spectrum
    plt.subplot(2, 1, 2)
    plt.plot(xf, np.abs(yf))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum of Euler Angle Magnitude')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the frequency analysis plot
    freq_figure_path = os.path.join(figures_dir, 'euler_magnitude_spectrum.png')
    plt.savefig(freq_figure_path, dpi=300)
    print(f"Frequency analysis saved to: {freq_figure_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
