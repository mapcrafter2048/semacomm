import pandas as pd
import matplotlib.pyplot as plt
import io

# Data provided by the user
data = """
LPIPS Alex: 0.724499, LPIPS VGG: 0.670124, PSNR: 5.138886, SSIM: 0.080097, Noise Level: -5
LPIPS Alex: 0.595418, LPIPS VGG: 0.605133, PSNR: 6.739927, SSIM: 0.110136, Noise Level: -3
LPIPS Alex: 0.523802, LPIPS VGG: 0.520158, PSNR: 8.652270, SSIM: 0.187675, Noise Level: -1
LPIPS Alex: 0.403440, LPIPS VGG: 0.405997, PSNR: 11.481908, SSIM: 0.350241, Noise Level: 1
LPIPS Alex: 0.254669, LPIPS VGG: 0.287526, PSNR: 14.196635, SSIM: 0.461908, Noise Level: 3
LPIPS Alex: 0.169361, LPIPS VGG: 0.207949, PSNR: 15.347704, SSIM: 0.502922, Noise Level: 5
LPIPS Alex: 0.151666, LPIPS VGG: 0.198187, PSNR: 15.727194, SSIM: 0.499297, Noise Level: 7
LPIPS Alex: 0.130182, LPIPS VGG: 0.160600, PSNR: 17.775057, SSIM: 0.499428, Noise Level: 9
"""


data2 = """
LPIPS Alex: 0.749207, LPIPS VGG: 0.709301, PSNR: 5.022845, SSIM: -16.88, Noise Level: -5
LPIPS Alex: 0.708900, LPIPS VGG: 0.679964, PSNR: 6.695361, SSIM: -13.97, Noise Level: -3
LPIPS Alex: 0.636882, LPIPS VGG: 0.575193, PSNR: 9.294808, SSIM: -11.88, Noise Level: -1
LPIPS Alex: 0.507371, LPIPS VGG: 0.488358, PSNR: 11.774474, SSIM: -9.59, Noise Level: 1
LPIPS Alex: 0.320560, LPIPS VGG: 0.350670, PSNR: 13.661627, SSIM: -7.05, Noise Level: 3
LPIPS Alex: 0.290537, LPIPS VGG: 0.306192, PSNR: 15.958559, SSIM: -5.02, Noise Level: 5
LPIPS Alex: 0.167854, LPIPS VGG: 0.227983, PSNR: 16.964282, SSIM: -3.63, Noise Level: 7
LPIPS Alex: 0.112235, LPIPS VGG: 0.148060, PSNR: 17.775057, SSIM: -3.86, Noise Level: 9
"""


# Parse the data into a list of dictionaries
parsed_data = []
for line in data.strip().split('\n'):
    entry = {}
    parts = line.split(', ')
    for part in parts:
        key, value = part.split(': ')
        entry[key.strip()] = float(value.strip())
    parsed_data.append(entry)

# Create DataFrame from the parsed data
df = pd.DataFrame(parsed_data)

# Sort DataFrame by 'Noise Level' to ensure correct plotting order for lines
df = df.sort_values(by='Noise Level').reset_index(drop=True)

# Define metrics to plot
metrics = ['LPIPS Alex', 'LPIPS VGG', 'PSNR', 'SSIM']



plt.figure(figsize=(10, 6))
plt.plot(df['Noise Level'], df['LPIPS Alex'], marker='o', linestyle='-', label='LPIPS Alex')
plt.plot(df['Noise Level'], df['LPIPS VGG'], marker='s', linestyle='-', label='LPIPS VGG')
plt.title('LPIPS vs. Channel SNR')
plt.xlabel('Channel SNR (Noise Level)')
plt.ylabel('LPIPS')
plt.legend()
plt.grid(True)
plt.show()


# Generate and display plots for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))
    # Plotting as a line graph with markers
    plt.plot(df['Noise Level'], df[metric], marker='o', linestyle='-')
    plt.title(f'{metric} vs. Channel SNR') # Set plot title as requested
    plt.xlabel('Channel SNR (Noise Level)') # Set x-axis label as requested
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()
