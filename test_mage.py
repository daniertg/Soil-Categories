import cv2
import numpy as np
import pickle
from scipy.interpolate import griddata

def load_model(model_path):
    """Load the trained model and label names"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['label_names']

def preprocess_image(image_bytes, N=80):
    """Preprocess a single image from raw bytes for prediction"""
    epsilon = 1e-8
    
    # Decode image from bytes
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Tidak dapat membaca gambar dari data bytes")
    
    # Compute FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    
    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Calculate radial profile
    def azimuthalAverage(image, center=None):
        y, x = np.indices(image.shape)
        if center is None:
            center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.hypot(x - center[0], y - center[1])
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]
        r_int = r_sorted.astype(int)
        deltar = r_int[1:] - r_int[:-1]
        rind = np.where(deltar)[0]
        nr = rind[1:] - rind[:-1]
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
        radial_prof = tbin / nr
        return radial_prof
    
    psd1D = azimuthalAverage(magnitude_spectrum)
    
    # Interpolate to fixed size
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')
    
    # Normalize
    interpolated = interpolated / (interpolated[0] + epsilon)
    
    return interpolated.reshape(1, -1)

def predict_soil_type(model, image_bytes, label_names):
    """Predict soil type for a new image"""
    # Preprocess image
    try:
        features = preprocess_image(image_bytes)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None
    
    probabilities = model.predict_proba(features)[0]
    
    # Get top 1 prediction with probability
    top_1_idx = np.argmax(probabilities)
    label = top_1_idx + 1  # Adjust index to match label numbers
    soil_type = label_names[label]
    prob = probabilities[top_1_idx]
    
    return [(soil_type, prob)]
