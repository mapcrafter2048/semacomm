import numpy as np

# SSIM values
ssim_values = [0.020492, 0.040049, 0.064837, 0.109864, 0.197259, 0.314935, 0.433501, 0.411190]


import numpy as np

def ssim_to_db(ssim_value, method='absolute'):
    """
    Convert SSIM value to decibels.
    
    Parameters:
    -----------
    ssim_value : float
        SSIM value (typically between -1 and 1)
    method : str
        'absolute' - uses abs(ssim_value)
        'shifted' - shifts SSIM to positive range
        'clamped' - clamps negative values to small positive value
    
    Returns:
    --------
    float : SSIM value in decibels
    """
    
    if method == 'absolute':
        # Take absolute value (most common approach)
        if ssim_value == 0:
            return float('-inf')  # or return a very large negative number
        return 10 * np.log10(abs(ssim_value))
    
    elif method == 'shifted':
        # Shift SSIM from [-1,1] to [0,2] range
        shifted_ssim = (ssim_value + 1) / 2
        if shifted_ssim <= 0:
            return float('-inf')
        return 10 * np.log10(shifted_ssim)
    
    elif method == 'clamped':
        # Clamp negative values to small positive value
        clamped_ssim = max(ssim_value, 1e-10)
        return 10 * np.log10(clamped_ssim)
    
    else:
        raise ValueError("Method must be 'absolute', 'shifted', or 'clamped'")


def ssim_to_db_simple(ssim_value):
    """Simple conversion of SSIM to dB using absolute value."""
    if abs(ssim_value) <= 1e-10:  # Handle very small values
        return -100  # Return a large negative dB value
    return -10 * np.log10(1- ssim_value)


if __name__ == "__main__":
# Convert each SSIM value to dB and print
    for i, ssim in enumerate(ssim_values):
        mssim_db = ssim_to_db_simple(ssim)
        print(f"SSIM {i+1}: {ssim} -> {mssim_db:.2f} dB")


