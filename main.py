# Importing Packages
# **********************************************************************************************************************************************************************************

import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import streamlit as st 

# Input 
# **********************************************************************************************************************************************************************************

def input(image, message):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption = message)

    return image

# All Feature Matrics Functions
# **************************************************************************************************************************************************************

def mean(image,flag,message):

    if flag == 1:
        mask_size = 2
        avg_total = []

        for c in range (0,image.shape[1]-mask_size+1):
            total = 0
            count = 0
            for r in range (0,image.shape[0]-mask_size+1):
                s = np.sum(image[r:r+mask_size,c:c+mask_size])
                d = mask_size * mask_size
                temp = s/d
                total = total + temp
                count = count + 1
            total = total / count
            avg_total.append(total)

        avg_total_chart = pd.DataFrame(avg_total)
        st.write(message)
        st.line_chart(avg_total_chart)
        

    total = 0
    avg_total = 0
    count = 0
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":  
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + float(image[r:r+1,c:c+1,0]) + float(image[r:r+1,c:c+1,1]) + float(image[r:r+1,c:c+1,2]) 
                count = count + 3

    else:
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + float(image[r:r+1,c:c+1])
                count = count + 1

    avg_total = total / count

    return avg_total

def std(image,flag,message):

    if flag == 1:
        mask_size = 2
        std_total = []

        for c in range (0,image.shape[1]-mask_size+1):
            total = 0
            count = 0
            for r in range (0,image.shape[0]-mask_size+1):
                s = np.sum(image[r:r+mask_size,c:c+mask_size])
                d = mask_size * mask_size
                avg = s/d
                temp2 = 0
                for i in range(0,mask_size):
                    for j in range(0,mask_size): 
                        t = image[r+i][c+j]
                        temp2 = temp2 + (t-avg)**2
                temp = np.sqrt(temp2/d)
                total = total + temp
                count = count + 1
            total = total / count
            std_total.append(total)

        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
            std_total_chart = pd.DataFrame(std_total,columns=["Blue", "Green", "Red"])
        else:
            std_total_chart = pd.DataFrame(std_total)

        st.write(message)
        st.line_chart(std_total_chart)
        

    std_total = 0.0
    total = 0.0
    count = 0
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":  
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1,0]) + float(image[r:r+1,c:c+1,1]) + float(image[r:r+1,c:c+1,2])) - avg_total)**2
                count = count + 3
        count = count - 3
    
    else:
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1])) - avg_total)**2
                count = count + 1
        count = count - 1

    std_total = np.sqrt(total/(count))
    std_total = std_total * 10000

    return std_total

def var(image,flag,message):

    if flag == 1:
        mask_size = 2
        var_total = []

        for c in range (0,image.shape[1]-mask_size+1):
            total = 0
            count = 0
            for r in range (0,image.shape[0]-mask_size+1):
                s = np.sum(image[r:r+mask_size,c:c+mask_size])
                d = mask_size * mask_size
                avg = s/d
                temp2 = 0
                for i in range(0,mask_size):
                    for j in range(0,mask_size): 
                        t = image[r+i][c+j]
                        temp2 = temp2 + (t-avg)**2
                temp = temp2/d
                total = total + temp
                count = count + 1
            total = total / count
            var_total.append(total)

        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
            var_total_chart = pd.DataFrame(var_total,columns=["Blue", "Green", "Red"])
        else:
            var_total_chart = pd.DataFrame(var_total)

        st.write(message)
        st.line_chart(var_total_chart)
        
        
    var_total = 0.0
    total = 0.0
    count = 0
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":  
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1,0]) + float(image[r:r+1,c:c+1,1]) + float(image[r:r+1,c:c+1,2])) - avg_total)**2
                count = count + 3
        count = count - 3
    
    else:
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1])) - avg_total)**2
                count = count + 1
        count = count - 1

    var_total = (total/(count)) * 10000
    
    return var_total

def rms(image,flag,message):

    if flag == 1:
        mask_size = 2
        rms_total = []

        for c in range (0,image.shape[1]-mask_size+1):
            total = 0
            count = 0
            for r in range (0,image.shape[0]-mask_size+1):
                s = np.sum(image[r:r+mask_size,c:c+mask_size])
                d = mask_size * mask_size
                avg = s/d
                temp2 = 0
                for i in range(0,mask_size):
                    for j in range(0,mask_size): 
                        t = image[r+i][c+j]
                        temp2 = temp2 + t*t
                temp = np.sqrt(temp2/d)
                total = total + temp
                count = count + 1
            total = total / count
            rms_total.append(total)
        
        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
            rms_total_chart = pd.DataFrame(rms_total,columns=["Blue", "Green", "Red"])
        else:
            rms_total_chart = pd.DataFrame(rms_total)

        st.write(message)
        st.line_chart(rms_total_chart)
        

    total = 0
    rms_total = 0
    total = 0
    count = 0

    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":  
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1,0]) + float(image[r:r+1,c:c+1,1]) + float(image[r:r+1,c:c+1,2])))**2
                count = count + 3

    else:
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + (float(image[r:r+1,c:c+1]))**2
                count = count + 1

    rms_total = np.sqrt(total/count)

    return rms_total

def skew(image):

    total = 0.0
    count = 0
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":  
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1,0]) + float(image[r:r+1,c:c+1,1]) + float(image[r:r+1,c:c+1,2])) - avg_total)**3
                count = count + 3
        count = count - 3
    
    else:
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total +((float(image[r:r+1,c:c+1])) - avg_total)**3
                count = count + 1
        count = count - 1

    d = (count * ((total)**3))

    skew_total = (total/d) * 10**30

    return skew_total

def kur(image):

    kur_total = 0.0
    total = 0.0
    count = 0
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":  
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1,0]) + float(image[r:r+1,c:c+1,1]) + float(image[r:r+1,c:c+1,2])) - avg_total)**4
                count = count + 3
        count = count - 3
    
    else:
        for c in range (0,image.shape[1]):
            for r in range (0,image.shape[0]):
                total = total + ((float(image[r:r+1,c:c+1])) - avg_total)**3
                count = count + 1
        count = count - 1

    d = (count * ((total)**4))

    kur_total = float(total/d) * 10**50

    return kur_total

def entropy(image):

    # Convert image to float for entropy calculation
    image = image.astype(float)
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection": 

        entropy_values = []
        for channel in range(image.shape[2]):
            # Calculate histogram for each channel
            hist, _ = np.histogram(image[:, :, channel].flatten(), bins=256, range=[0, 256])

            # Normalize histogram to get probabilities
            probabilities = hist / np.sum(hist)

            # Calculate entropy for each channel
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to avoid log(0)
            entropy_values.append(entropy)

        # Return the average entropy
        entropy_total = np.mean(entropy_values)

    else:
        # Calculate histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])

        # Normalize histogram to get probabilities
        probabilities = hist / np.sum(hist)

        # Calculate entropy
        entropy_total = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to avoid log(0)

    return entropy_total

def histogram(image,flag):

    if flag == 1:

        st.subheader("Histogram of each channel")
        
        # Split the image into its color channels
        b, g, r = cv2.split(image)

        # Compute histograms for each channel
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        # Create a figure for the histograms
        fig, ax = plt.subplots()

        # Plot all histograms together
        ax.plot(hist_b, color='blue', label='Blue')
        ax.plot(hist_g, color='green', label='Green')
        ax.plot(hist_r, color='red', label='Red')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Color Histograms')
        ax.legend()

        # Display the figure in Streamlit
        st.pyplot(fig)

        st.subheader("Histogram Cumulative")
        
        hist = cv2.calcHist([image],[2],None,[256],[0,500])
        st.line_chart(hist)

    else:
        st.subheader("Histogram")
        
        hist = cv2.calcHist([image],[0],None,[256],[0,500])
        st.line_chart(hist)

def histogram_mean(image):

    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
        # Split the image into its color channels
        b, g, r = cv2.split(image)
        add_b = 0
        add_g = 0
        add_r = 0
        # Compute histograms for each channel
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        for i in range(0,256):
            add_b = add_b + (float(hist_b[i])*i)
            add_g = add_g + (float(hist_g[i])*i)
            add_r = add_r + (float(hist_r[i])*i)

        add_total = (add_r + add_g + add_b) / 256

    else:

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        add = 0
        for i in range(0,256):
            add = add + float(hist[i]) * i
        add_total  = add / 256

    return add_total

def object(image,x,y,width,height,message):

    cut = image
    edges = cv2.Canny(cut,100,200)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (x,y,width,height)
    cv2.grabCut(cut,edges,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((edges==2)|(edges==0),0,1).astype('uint8')
    img = cut*mask[:,:,np.newaxis]
    # st.subheader("Object Image")
    # st.image(img, caption = "Object Image After GrabCut")
    # st.write("Image shape:", img.shape)

    cut = image

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    binary_message = "Binary " + message
    st.image(binary_image, caption = binary_message)
    # st.write("Image shape:", binary_image.shape)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    final = cv2.bitwise_and(cut, cut, mask=closing)
    
    morphological_message = "Morphological " + message
    st.image(closing, caption = morphological_message)
    # st.write("Image shape:", closing.shape)
    st.image(final, caption = message)
    # st.write("Image shape:", final.shape)
    return final

# Transformation
# *************************************************************************************************************************************************************

def notransformation(image,message):

    st.image(image, caption = message)

    return image

def gamma(image,gamma_value,message):

    # Define the gamma value (adjust as needed)
    gamma = gamma_value

    # Perform gamma correction
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    st.image(gamma_corrected, caption = message)

    return gamma_corrected

def log(image, message):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply log transformation to each color channel
    c = 1  # Constant value to avoid log(0)
    log_transformed_b = c * np.log1p(b.astype(np.float32))
    log_transformed_g = c * np.log1p(g.astype(np.float32))
    log_transformed_r = c * np.log1p(r.astype(np.float32))

    # Scale the values to 0-255 range
    log_transformed_b = cv2.normalize(log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    log_transformed_g = cv2.normalize(log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    log_transformed_r = cv2.normalize(log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Merge the log-transformed color channels back into an image
    log_transformed_image = cv2.merge((log_transformed_b, log_transformed_g, log_transformed_r))

    st.image(log_transformed_image, caption = message)

    return log_transformed_image

def inverselog(image, message):
    # Split the original image into its color channels
    b, g, r = cv2.split(image)

    # Apply log transformation to each color channel
    c = 1  # Constant value to avoid log(0)
    log_transformed_b = c * np.log1p(b.astype(np.float32))
    log_transformed_g = c * np.log1p(g.astype(np.float32))
    log_transformed_r = c * np.log1p(r.astype(np.float32))

    # Apply inverse log transformation to each color channel
    inv_log_transformed_b = np.expm1(log_transformed_b)
    inv_log_transformed_g = np.expm1(log_transformed_g)
    inv_log_transformed_r = np.expm1(log_transformed_r)

    # Scale the values to 0-255 range
    inv_log_transformed_b = cv2.normalize(inv_log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    inv_log_transformed_g = cv2.normalize(inv_log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    inv_log_transformed_r = cv2.normalize(inv_log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Merge the inverse log-transformed color channels back into an image
    inv_log_transformed_image = cv2.merge((inv_log_transformed_b, inv_log_transformed_g, inv_log_transformed_r))

    # Display the image
    st.image(inv_log_transformed_image, caption = message)

    return inv_log_transformed_image

# Coloration
# *************************************************************************************************************************************************************

def nocoloration(image, message):

    st.image(image, caption = message)

    return image

def gray(image, message):

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    st.image(gray_image, caption = message)

    return gray_image

def hue(image, message):
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract the hue channel
    hue_image = hsv_image[:, :, 0]  # Hue channel is the first channel in HSV

    # Display the hue image
    st.image(hue_image, caption = message, channels='HSV', use_column_width=True)

    return hue_image

def pseudo_spring(image,message):

    pseudo_spring_image = cv2.applyColorMap(image, cv2.COLORMAP_SPRING)
    st.image(pseudo_spring_image, caption = message)

    return pseudo_spring_image

def pseudo_hot(image,message):

    pseudo_hot_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    st.image(pseudo_hot_image, caption = message)

    return pseudo_hot_image

def pseudo_cool(image,message):

    pseudo_cool_image = cv2.applyColorMap(image, cv2.COLORMAP_COOL)
    st.image(pseudo_cool_image, caption = message)

    return pseudo_cool_image

def pseudo_rainbow(image,message):

    pseudo_rainbow_image = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)
    st.image(pseudo_rainbow_image, caption = message)

    return pseudo_rainbow_image

def pseudo_hsv(image,message):

    pseudo_hsv_image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
    st.image(pseudo_hsv_image, caption = message)

    return pseudo_hsv_image

def pseudo_jet(image,message):

    pseudo_jet_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    st.image(pseudo_jet_image, caption = message)

    return pseudo_jet_image

# All Edge Detection Functions
# **************************************************************************************************************************************************************

def noedge(image, message):

    st.image(image, caption = message)
    return image

def canny(image,lower,upper,message):

    edges = cv2.Canny(image,lower,upper)
    st.image(edges, caption = message)
    return edges

def otsu(image,message):
    edges = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    st.image(edges, caption = message)
    return edges

def prewitt(image,message):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    edges = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
    st.image(edges, caption = message)
    return edges

def robert(image,message):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    img_robertx = cv2.filter2D(image, -1, kernelx)
    img_roberty = cv2.filter2D(image, -1, kernely)
    edges = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)
    st.image(edges, caption = message)
    return edges

# Streamlit Simulation
# **************************************************************************************************************************************************************

# Title
st.title("Simulation for Image Processing Operations")

# Sidebar
with st.sidebar:
    # choice_img = st.selectbox("Image Set", ["Select one", "Set1", "Set2", "Set3", "Set4", "Set5", "Set6"])
    choice_img = st.file_uploader("Input Images: MAX 9 IMAGES", type= ['bmp','jpg','jpeg','png'], accept_multiple_files=True)

    if len(choice_img) >= 1:
        on5 = st.toggle('Original Image Histogram')
        choice7 = st.selectbox("Operations on", ["Select one", "Object With Background", "Object Without Background"])

        if len(choice_img) >= 1:
            img1 = cv2.imdecode(np.frombuffer(choice_img[0].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 4:
            img2 = cv2.imdecode(np.frombuffer(choice_img[3].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 7:
            img3 = cv2.imdecode(np.frombuffer(choice_img[6].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 2:
            img4 = cv2.imdecode(np.frombuffer(choice_img[1].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 5:
            img5 = cv2.imdecode(np.frombuffer(choice_img[4].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 8:
            img6 = cv2.imdecode(np.frombuffer(choice_img[7].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 3:
            img7 = cv2.imdecode(np.frombuffer(choice_img[2].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 6:
            img8 = cv2.imdecode(np.frombuffer(choice_img[5].read(), np.uint8), cv2.IMREAD_COLOR)
        if len(choice_img) >= 9:
            img9 = cv2.imdecode(np.frombuffer(choice_img[8].read(), np.uint8), cv2.IMREAD_COLOR)

        if choice7 != "Select one":
            if choice7 == "Object Without Background":

                # if len(choice_img) >= 9:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3", "Image4", "Image5", "Image6", "Image7", "Image8", "Image9"])
                # elif len(choice_img) >= 8:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3", "Image4", "Image5", "Image6", "Image7", "Image8"])
                # elif len(choice_img) >= 7:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3", "Image4", "Image5", "Image6", "Image7"])
                # elif len(choice_img) >= 6:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3", "Image4", "Image5", "Image6"])
                # elif len(choice_img) >= 5:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3", "Image4", "Image5"])
                # elif len(choice_img) >= 4:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3", "Image4"])
                # elif len(choice_img) >= 3:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2", "Image3"])
                # elif len(choice_img) >= 2:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1", "Image2"])
                # elif len(choice_img) >= 1:
                #     choice8 = st.selectbox("Choose GrabCut Parameters", ["Select one", "Image1"])

                on1 = st.toggle('Original Object Detected Image Histogram')
            choice6 = st.selectbox("Image Transformation", ["Select one", "No Transformation", "Gamma Transformation", "Log Transformation", "Inverse Log Transformation"])

            if choice6 != "Select one":
                on2 = st.toggle('Transformed Image Histogram')
                choice4 = st.selectbox("Image Coloration", ["Select one", "No Coloration Image", "Gray Coloration", "HUE Coloration", "Pseudo Coloration"])

                if choice4 != "Select one":

                    if choice4 == "Pseudo Coloration":
                        choice5 = st.selectbox("Types of Pseudo Coloration", ["Select one", "Spring", "Hot", "Cool", "Rainbow", "HSV", "JET"])
                        if choice5 != "Select one":
                            on3 = st.toggle('Coloration Image Histogram')
                            choice2 = st.selectbox("Edge Detection", ["Select one", "No Edge Detection", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])

                    else:
                        on3 = st.toggle('Coloration Image Histogram')
                        if choice4 == "HUE Coloration":
                            choice2 = st.selectbox("Edge Detection", ["Select one", "No Edge Detection", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                            
                        else:
                            choice2 = st.selectbox("Edge Detection", ["Select one", "No Edge Detection", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                    
                    if choice4 == "Pseudo Coloration" and choice5 == "Select one":
                        pass

                    elif choice2 != 'Select one':

                        if choice2 == "No Edge Detection":
                            on4 = st.toggle('Edge Detection Histogram')
                            on6 = st.toggle("Features: Go to 'Summary Table' tab")
                            if on6:
                                on7 = st.toggle("With graph: Go to 'Graph' tab")
                        else:
                            choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"]) 
                            if choice3 != "Select one":
                                on4 = st.toggle('Edge Detection Histogram')
                                on6 = st.toggle("Features: Go to 'Summary Table' tab")
                                if on6:
                                    on7 = st.toggle("With graph: Go to 'Graph' tab")
                    
tab1, tab2, tab3 = st.tabs(["Home", "Summary Table", "Graph"])
# All Options 
# **************************************************************************************************************************************************************

with tab1:
    if len(choice_img) >= 1:

        message1 = "Image 1"
        message2 = "Image 4"
        message3 = "Image 7"
        message4 = "Image 2"
        message5 = "Image 5" 
        message6 = "Image 8"
        message7 = "Image 3"
        message8 = "Image 6"
        message9 = "Image 9"

        # st.header('', divider='rainbow')

        with st.expander("Original Image"):
            # Creating of Columns                       
            col1, col2, col3 = st.columns([1,1,1])
            with col1:

                if len(choice_img) >= 1:

                    st.header("Original Image")
                    #img1 = cv2.imdecode(np.frombuffer(choice_img[0].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image1 = input(img1,message1)
                    img1 = original_image1 

                if len(choice_img) >= 4:

                    #img2 = cv2.imdecode(np.frombuffer(choice_img[3].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image2 = input(img2,message2)
                    img2 = original_image2

                if len(choice_img) >= 7:

                    #img3 = cv2.imdecode(np.frombuffer(choice_img[6].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image3 = input(img3,message3)
                    img3 = original_image3

            with col2:

                if len(choice_img) >= 2:

                    st.header("Original Image")
                    #img4 = cv2.imdecode(np.frombuffer(choice_img[1].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image4 = input(img4,message4)
                    img4 = original_image4 

                if len(choice_img) >= 5:

                    #img5 = cv2.imdecode(np.frombuffer(choice_img[4].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image5 = input(img5,message5)
                    img5 = original_image5

                if len(choice_img) >= 8:

                    #img6 = cv2.imdecode(np.frombuffer(choice_img[7].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image6 = input(img6,message6)
                    img6 = original_image6
                    
            with col3:

                if len(choice_img) >= 3:

                    st.header("Original Image")
                    #img7 = cv2.imdecode(np.frombuffer(choice_img[2].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image7 = input(img7,message7)
                    img7 = original_image7

                if len(choice_img) >= 6:

                    #img8 = cv2.imdecode(np.frombuffer(choice_img[5].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image8 = input(img8,message8)
                    img8 = original_image8

                if len(choice_img) >= 9:

                    #img9 = cv2.imdecode(np.frombuffer(choice_img[8].read(), np.uint8), cv2.IMREAD_COLOR)
                    original_image9 = input(img9,message9)
                    img9 = original_image9

            if on5:

                with col1:
                    if len(choice_img) >= 1:
                        histogram(img1,1)
                    if len(choice_img) >= 4:
                        histogram(img2,1)
                    if len(choice_img) >= 7:
                        histogram(img3,1)
                with col2:
                    if len(choice_img) >= 2:
                        histogram(img4,1)
                    if len(choice_img) >= 5:
                        histogram(img5,1)
                    if len(choice_img) >= 8:
                        histogram(img6,1)
                with col3:
                    if len(choice_img) >= 3:
                        histogram(img7,1)
                    if len(choice_img) >= 6:
                        histogram(img8,1)
                    if len(choice_img) >= 9:
                        histogram(img9,1)
        
        if choice7 != "Select one":

            # if choice7 == "Object With Background":
            #     pass

            if choice7 == "Object Without Background":

                with st.expander("Object Image"):
                        # Creating of Columns                       
                        col1, col2, col3 = st.columns([1,1,1])
                        with col1:

                            if len(choice_img) >= 1:
                                st.header("Object Image")

                                cut1 = original_image1
                                x1 = st.slider("Horizontal top-left corner:",0,original_image1.shape[0], value = 0)
                                y1 = st.slider("Vertical top-left corner:",0,original_image1.shape[1], value = 0)
                                w1 = st.slider("Horizontal width:",0,original_image1.shape[1], value = original_image1.shape[1] -1)
                                h1 = st.slider("Vertical height:",0,original_image1.shape[0], value = original_image1.shape[0])
                                img1 = object(cut1,x1,y1,w1,h1,message1)

                            if len(choice_img) >= 4:
                                cut2 = original_image2
                                x2 = st.slider("Horizontal top-left corner:",0,original_image2.shape[0], value = 0)
                                y2 = st.slider("Vertical top-left corner:",0,original_image2.shape[1], value = 0)
                                w2 = st.slider("Horizontal width:",0,original_image2.shape[1], value = original_image2.shape[1] -1)
                                h2 = st.slider("Vertical height:",0,original_image2.shape[0], value = original_image2.shape[0])
                                img2 = object(cut2,x2, y2, w2, h2,message2)
                            
                            if len(choice_img) >= 7:
                                cut3 = original_image3
                                x3 = st.slider("Horizontal top-left corner:",0,original_image3.shape[0], value = 0)
                                y3 = st.slider("Vertical top-left corner:",0,original_image3.shape[1], value = 0)
                                w3 = st.slider("Horizontal width:",0,original_image3.shape[1], value = original_image3.shape[1] -1)
                                h3 = st.slider("Vertical height:",0,original_image3.shape[0], value = original_image3.shape[0])
                                img3 = object(cut3,x3, y3, w3, h3,message3)

                        with col2:

                            if len(choice_img) >= 2:
                                st.header("Object Image")

                                cut4 = original_image4
                                x4 = st.slider("Horizontal top-left corner:",0,original_image4.shape[0], value = 0)
                                y4 = st.slider("Vertical top-left corner:",0,original_image4.shape[1], value = 0)
                                w4 = st.slider("Horizontal width:",0,original_image4.shape[1], value = original_image4.shape[1] -1)
                                h4 = st.slider("Vertical height:",0,original_image4.shape[0], value = original_image4.shape[0])
                                img4 = object(cut4,x4, y4, w4, h4,message4)

                            if len(choice_img) >= 5:
                                cut5 = original_image5
                                x5 = st.slider("Horizontal top-left corner:",0,original_image5.shape[0], value = 0)
                                y5 = st.slider("Vertical top-left corner:",0,original_image5.shape[1], value = 0)
                                w5 = st.slider("Horizontal width:",0,original_image5.shape[1], value = original_image5.shape[1] -1)
                                h5 = st.slider("Vertical height:",0,original_image5.shape[0], value = original_image5.shape[0])
                                img5 = object(cut5,x5, y5, w5, h5,message5)

                            if len(choice_img) >= 8:
                                cut6 = original_image6
                                x6 = st.slider("Horizontal top-left corner:",0,original_image6.shape[0], value = 0)
                                y6 = st.slider("Vertical top-left corner:",0,original_image6.shape[1], value = 0)
                                w6 = st.slider("Horizontal width:",0,original_image6.shape[1], value = original_image6.shape[1] -1)
                                h6 = st.slider("Vertical height:",0,original_image6.shape[0], value = original_image6.shape[0])
                                img6 = object(cut6,x6, y6, w6, h6,message6)

                        with col3:

                            if len(choice_img) >= 3:
                                st.header("Object Image")

                                cut7 = original_image7
                                x7 = st.slider("Horizontal top-left corner:",0,original_image7.shape[0], value = 0)
                                y7 = st.slider("Vertical top-left corner:",0,original_image7.shape[1], value = 0)
                                w7 = st.slider("Horizontal width:",0,original_image7.shape[1], value = original_image7.shape[1] -1)
                                h7 = st.slider("Vertical height:",0,original_image7.shape[0], value = original_image7.shape[0])
                                img7 = object(cut7,x7, y7, w7, h7,message7)

                            if len(choice_img) >= 6:
                                cut8 = original_image8
                                x8 = st.slider("Horizontal top-left corner:",0,original_image8.shape[0], value = 0)
                                y8 = st.slider("Vertical top-left corner:",0,original_image8.shape[1], value = 0)
                                w8 = st.slider("Horizontal width:",0,original_image8.shape[1], value = original_image8.shape[1] -1)
                                h8 = st.slider("Vertical height:",0,original_image8.shape[0], value = original_image8.shape[0])
                                img8 = object(cut8,x8, y8, w8, h8,message8)

                            if len(choice_img) >= 9:
                                cut9 = original_image9
                                x9 = st.slider("Horizontal top-left corner:",0,original_image9.shape[0], value = 0)
                                y9 = st.slider("Vertical top-left corner:",0,original_image9.shape[1], value = 0)
                                w9 = st.slider("Horizontal width:",0,original_image9.shape[1], value = original_image9.shape[1] -1)
                                h9 = st.slider("Vertical height :",0,original_image9.shape[0], value = original_image9.shape[0])
                                img9 = object(cut9,x9, y9, w9, h9,message9)

                        if on1:
                            with col1:
                                if len(choice_img) >= 1:
                                    histogram(img1,1)
                                if len(choice_img) >= 4:
                                    histogram(img2,1)
                                if len(choice_img) >= 7:
                                    histogram(img3,1)
                            with col2:
                                if len(choice_img) >= 2:
                                    histogram(img4,1)
                                if len(choice_img) >= 5:
                                    histogram(img5,1)
                                if len(choice_img) >= 8:
                                    histogram(img6,1)
                            with col3:
                                if len(choice_img) >= 3:
                                    histogram(img7,1)
                                if len(choice_img) >= 6:
                                    histogram(img8,1)
                                if len(choice_img) >= 9:
                                    histogram(img9,1)
        
            if choice6 != "Select one":

                with st.expander("Image Transformation"):
                    # Creating of Columns                       
                    col1, col2, col3 = st.columns([1,1,1])

                    # Image Transformation
                    # **************************************************************************************************************************************************************

                    if choice6 == "No Transformation":
                        with col1:

                            if len(choice_img) >= 1:
                                st.header("No Transformation Image")
                                img1 = notransformation(img1, message1)
                                transformed_image1 = img1

                            if len(choice_img) >= 4:
                                img2 = notransformation(img2, message2)
                                transformed_image2 = img2

                            if len(choice_img) >= 7:
                                img3 = notransformation(img3, message3)
                                transformed_image3 = img3

                        with col2:

                            if len(choice_img) >= 2:
                                st.header("No Transformation Image")

                                img4 = notransformation(img4, message4)
                                transformed_image4 = img4

                            if len(choice_img) >= 5:
                                img5 = notransformation(img5, message5)
                                transformed_image5 = img5

                            if len(choice_img) >= 8:
                                img6 = notransformation(img6, message6)
                                transformed_image6 = img6

                        with col3:

                            if len(choice_img) >= 3:
                                st.header("No Transformation Image")

                                img7 = notransformation(img7, message7)
                                transformed_image7 = img7

                            if len(choice_img) >= 6:
                                img8 = notransformation(img8, message8)
                                transformed_image8 = img8

                            if len(choice_img) >= 9:
                                img1 = notransformation(img9, message9)
                                transformed_image9 = img9

                    if choice6 == "Gamma Transformation":

                        with col1:

                            if len(choice_img) >= 1:

                                st.header("Gamma Transformation Image")

                                gamma_value1 = st.slider("Gamma Value of Image 1:",0.0,20.0, value = 0.1)
                                gamma_corrected1 = gamma(img1, gamma_value1, message1)
                                transformed_image1 = gamma_corrected1

                            if len(choice_img) >= 4:
                                gamma_value2 = st.slider("Gamma Value of Image 4:",0.0,20.0, value = 0.1)
                                gamma_corrected2 = gamma(img2, gamma_value2, message2)
                                transformed_image2 = gamma_corrected2

                            if len(choice_img) >= 7:
                                gamma_value3 = st.slider("Gamma Value of Image 7:",0.0,20.0, value = 0.1)
                                gamma_corrected3 = gamma(img3, gamma_value3, message3)
                                transformed_image3 = gamma_corrected3

                        with col2:

                            if len(choice_img) >= 2:
                                st.header("Gamma Transformation Image")

                                gamma_value4 = st.slider("Gamma Value of Image 2:",0.0,20.0, value = 0.1)
                                gamma_corrected4 = gamma(img4, gamma_value4, message4)
                                transformed_image4 = gamma_corrected4

                            if len(choice_img) >= 5:
                                gamma_value5 = st.slider("Gamma Value of Image 5:",0.0,20.0, value = 0.1)
                                gamma_corrected5 = gamma(img5, gamma_value5, message5)
                                transformed_image5 = gamma_corrected5

                            if len(choice_img) >= 8:
                                gamma_value6 = st.slider("Gamma Value of Image 8:",0.0,20.0, value = 0.1)
                                gamma_corrected6 = gamma(img6, gamma_value6, message6)
                                transformed_image6 = gamma_corrected6
                            

                        with col3:

                            if len(choice_img) >= 3:
                                st.header("Gamma Transformation Image")

                                gamma_value7 = st.slider("Gamma Value of Image 3:",0.0,20.0, value = 0.1)
                                gamma_corrected7 = gamma(img7, gamma_value7, message7)
                                transformed_image7 = gamma_corrected7
                                
                            if len(choice_img) >= 6:
                                gamma_value8 = st.slider("Gamma Value of Image 6:",0.0,20.0, value = 0.1)
                                gamma_corrected8 = gamma(img8, gamma_value8, message8)
                                transformed_image8 = gamma_corrected8
                                
                            if len(choice_img) >= 9:
                                gamma_value9 = st.slider("Gamma Value of Image 9:",0.0,20.0, value = 0.1)
                                gamma_corrected9 = gamma(img9, gamma_value9, message9)
                                transformed_image9 = gamma_corrected9
                                            
                    if choice6 == "Log Transformation":

                        with col1:

                            if len(choice_img) >= 1:
                                st.header("Log Transformation Image")

                                log_transformed_image1 = log(img1, message1)          
                                transformed_image1 = log_transformed_image1
                            
                            if len(choice_img) >= 4:
                                log_transformed_image2 = log(img2, message2)          
                                transformed_image2 = log_transformed_image2
                                
                            if len(choice_img) >= 7:
                                log_transformed_image3 = log(img3, message3)          
                                transformed_image3 = log_transformed_image3

                        with col2:

                            if len(choice_img) >= 2:
                                st.header("Log Transformation Image")
                                
                                log_transformed_image4 = log(img4, message4)
                                transformed_image4 = log_transformed_image4
                                
                            if len(choice_img) >= 5:
                                log_transformed_image5 = log(img5, message5)          
                                transformed_image5 = log_transformed_image5
                                
                            if len(choice_img) >= 8:
                                log_transformed_image6 = log(img6, message6)          
                                transformed_image6 = log_transformed_image6

                        with col3:

                            if len(choice_img) >= 3:
                                st.header("Log Transformation Image")
                                
                                log_transformed_image7 = log(img7, message7)
                                transformed_image7 = log_transformed_image7
                
                            if len(choice_img) >= 6:
                                log_transformed_image8 = log(img8, message8)          
                                transformed_image8 = log_transformed_image8
                                
                            if len(choice_img) >= 9:
                                log_transformed_image9 = log(original_image9,message9)          
                                transformed_image9 = log_transformed_image9
                            
                    if choice6 == "Inverse Log Transformation":
                        
                        with col1:

                            if len(choice_img) >= 1:
                                st.header("Inverse Log Transformation Image")

                                inv_log_transformed_image1 = inverselog(img1, message1)
                                transformed_image1 = inv_log_transformed_image1
                                
                            if len(choice_img) >= 4:
                                inv_log_transformed_image2 = inverselog(img2, message2)
                                transformed_image2 = inv_log_transformed_image2
                                
                            if len(choice_img) >= 7:
                                inv_log_transformed_image3 = inverselog(img3, message3)
                                transformed_image3 = inv_log_transformed_image3
                            

                        with col2:

                            if len(choice_img) >= 2:
                                st.header("Inverse Log Transformation Image")
                                
                                inv_log_transformed_image4 = inverselog(img4,message4)
                                transformed_image4 = inv_log_transformed_image4
                                
                            if len(choice_img) >= 5:
                                inv_log_transformed_image5 = inverselog(img5,message5)
                                transformed_image5 = inv_log_transformed_image5
                                
                            if len(choice_img) >= 8:
                                inv_log_transformed_image6 = inverselog(img6,message6)
                                transformed_image6 = inv_log_transformed_image6
                            

                        with col3:

                            if len(choice_img) >= 3:
                                st.header("Inverse Log Transformation Image")
                                
                                inv_log_transformed_image7 = inverselog(img7,message7)
                                transformed_image7 = inv_log_transformed_image7
                                
                            if len(choice_img) >= 6:
                                inv_log_transformed_image8 = inverselog(img8,message8)
                                transformed_image8 = inv_log_transformed_image8
                                
                            if len(choice_img) >= 9:
                                inv_log_transformed_image9 = inverselog(img9,message9)
                                transformed_image9 = inv_log_transformed_image9
                            
                    if on2:
                        with col1:
                            if len(choice_img) >= 1:
                                histogram(transformed_image1,1)
                            if len(choice_img) >= 4:
                                histogram(transformed_image2,1)
                            if len(choice_img) >= 7:
                                histogram(transformed_image3,1)
                        with col2:
                            if len(choice_img) >= 2:
                                histogram(transformed_image4,1)
                            if len(choice_img) >= 5:
                                histogram(transformed_image5,1)
                            if len(choice_img) >= 8:
                                histogram(transformed_image6,1)
                        with col3:
                            if len(choice_img) >= 3:
                                histogram(transformed_image7,1)
                            if len(choice_img) >= 6:
                                histogram(transformed_image8,1)
                            if len(choice_img) >= 9:
                                histogram(transformed_image9,1)
                            
                    # Image Coloration
                    # **************************************************************************************************************************************************************

            if choice6 != "Select one":

                if choice4 != "Select one":

                    with st.expander("Image Coloration"):
                        # Creating of Columns                       
                        col1, col2, col3 = st.columns([1,1,1])

                        if choice4 == "No Coloration Image":

                            with col1:
                                
                                if len(choice_img) >= 1:
                                    st.header("No Coloration Image")

                                    transformed_image1 = nocoloration(transformed_image1, message1)
                                    apply_image1 = transformed_image1
                                    
                                if len(choice_img) >= 4:
                                    transformed_image2 = nocoloration(transformed_image2, message2)
                                    apply_image2 = transformed_image2
                                    
                                if len(choice_img) >= 7:
                                    transformed_image3 = nocoloration(transformed_image3, message3)
                                    apply_image3 = transformed_image3
                                

                            with col2:
                                if len(choice_img) >= 2:
                                    st.header("No Coloration Image")

                                    transformed_image4 = nocoloration(transformed_image4, message4)
                                    apply_image4 = transformed_image4
                                    
                                if len(choice_img) >= 5:    
                                    transformed_image5 = nocoloration(transformed_image5, message5)
                                    apply_image5 = transformed_image5
                                    
                                if len(choice_img) >= 8:
                                    transformed_image6 = nocoloration(transformed_image6, message6)
                                    apply_image6 = transformed_image6
                                
                            with col3:
                                if len(choice_img) >= 3:
                                    st.header("No Coloration Image")

                                    transformed_image7 = nocoloration(transformed_image7, message7)
                                    apply_image7 = transformed_image7
                                    
                                if len(choice_img) >= 6:
                                    transformed_image8 = nocoloration(transformed_image8, message8)
                                    apply_image8 = transformed_image8
                                    
                                if len(choice_img) >= 9:
                                    transformed_image9 = nocoloration(transformed_image9, message9)
                                    apply_image9 = transformed_image9
                                
                        if choice4 == "Gray Coloration":

                            with col1:

                                if len(choice_img) >= 1:
                                    st.header("Gray Coloration Image")

                                    gray_image1 = gray(transformed_image1, message1)
                                    apply_image1 = gray_image1
                                    
                                if len(choice_img) >= 4:
                                    gray_image2 = gray(transformed_image2, message2)
                                    apply_image2 = gray_image2
                                    
                                if len(choice_img) >= 7:
                                    gray_image3 = gray(transformed_image3, message3)
                                    apply_image3 = gray_image3
                                

                            with col2:
                                
                                if len(choice_img) >= 2:
                                    st.header("Gray Coloration Image")

                                    gray_image4 = gray(transformed_image4, message4)
                                    apply_image4 = gray_image4
                                    
                                if len(choice_img) >= 5:
                                    gray_image5 = gray(transformed_image5, message5)
                                    apply_image5 = gray_image5
                                    
                                if len(choice_img) >= 8:
                                    gray_image6 = gray(transformed_image6, message6)
                                    apply_image6 = gray_image6
                                

                            with col3:

                                if len(choice_img) >= 3:
                                    st.header("Gray Coloration Image")

                                    gray_image7 = gray(transformed_image7, message7)
                                    apply_image7 = gray_image7
                                    
                                if len(choice_img) >= 6:
                                    gray_image8 = gray(transformed_image8, message8)
                                    apply_image8 = gray_image8
                                    
                                if len(choice_img) >= 9:
                                    gray_image9 = gray(transformed_image9, message9)
                                    apply_image9 = gray_image9

                        if choice4 == "HUE Coloration":
                                
                            with col1:
                                
                                if len(choice_img) >= 1:
                                    st.header("HUE Coloration Image")

                                    hue_image1 = hue(transformed_image1, message1)
                                    apply_image1 = hue_image1
                                    
                                if len(choice_img) >= 4:
                                    hue_image2 = hue(transformed_image2,message2)
                                    apply_image2 = hue_image2
                                    
                                if len(choice_img) >= 7:
                                    hue_image3 = hue(transformed_image3, message3)
                                    apply_image3 = hue_image3
                                

                            with col2:

                                if len(choice_img) >= 2:
                                    st.header("HUE Coloration Image")

                                    hue_image4 = hue(transformed_image4, message4)
                                    apply_image4 = hue_image4
                                    
                                if len(choice_img) >= 5:
                                    hue_image5 = hue(transformed_image5, message5)
                                    apply_image5 = hue_image5
                                    
                                if len(choice_img) >= 8:
                                    hue_image6 = hue(transformed_image6, message6)
                                    apply_image6 = hue_image6
                                

                            with col3:

                                if len(choice_img) >= 3:
                                    st.header("HUE Coloration Image")
                                
                                    hue_image7 = hue(transformed_image7, message7)
                                    apply_image7 = hue_image7
                                    
                                if len(choice_img) >= 6:
                                    hue_image8 = hue(transformed_image8, message8)
                                    apply_image8 = hue_image8
                                    
                                if len(choice_img) >= 9:
                                    hue_image9 = hue(transformed_image9, message9)
                                    apply_image9 = hue_image9

                        if choice4 == "Pseudo Coloration":

                            if choice5 != "Select one":

                                if choice5 == "Spring":

                                    with col1:

                                        if len(choice_img) >= 1:
                                            st.header("Pseudo Spring Coloration Image")

                                            pseudo_spring_image1 = pseudo_spring(transformed_image1, message1)
                                            apply_image1 = pseudo_spring_image1
                                            
                                        if len(choice_img) >= 4:
                                            pseudo_spring_image2 = pseudo_spring(transformed_image2, message2)
                                            apply_image2 = pseudo_spring_image2
                                            
                                        if len(choice_img) >= 7:
                                            pseudo_spring_image3 = pseudo_spring(transformed_image3, message3)
                                            apply_image3 = pseudo_spring_image3
                                        

                                    with col2:

                                        if len(choice_img) >= 2:
                                            st.header("Pseudo Spring Coloration Image")

                                            pseudo_spring_image4 = pseudo_spring(transformed_image4, message4)
                                            apply_image4 = pseudo_spring_image4
                                            
                                        if len(choice_img) >= 5:
                                            pseudo_spring_image5 = pseudo_spring(transformed_image5, message5)
                                            apply_image5 = pseudo_spring_image5
                                            
                                        if len(choice_img) >= 8:
                                            pseudo_spring_image6 = pseudo_spring(transformed_image6, message6)
                                            apply_image6 = pseudo_spring_image6
                                        

                                    with col3:

                                        if len(choice_img) >= 3:
                                            st.header("Pseudo Spring Coloration Image")
                                        
                                            pseudo_spring_image7 = pseudo_spring(transformed_image7, message7)
                                            apply_image7 = pseudo_spring_image7
                                            
                                        if len(choice_img) >= 6:
                                            pseudo_spring_image8 = pseudo_spring(transformed_image8, message8)
                                            apply_image8 = pseudo_spring_image8
                                            
                                        if len(choice_img) >= 9:
                                            pseudo_spring_image9 = pseudo_spring(transformed_image9, message9)
                                            apply_image9 = pseudo_spring_image9
                                        
                                if choice5 == "Hot":

                                    with col1:

                                        if len(choice_img) >= 1:
                                            st.header("Pseudo Hot Coloration Image")

                                            pseudo_hot_image1 = pseudo_hot(transformed_image1, message1)
                                            apply_image1 = pseudo_hot_image1
                                            
                                        if len(choice_img) >= 4:
                                            pseudo_hot_image2 = pseudo_hot(transformed_image2, message2)
                                            apply_image2 = pseudo_hot_image2
                                            
                                        if len(choice_img) >= 7:
                                            pseudo_hot_image3 = pseudo_hot(transformed_image3, message3)
                                            apply_image3 = pseudo_hot_image3
                                        

                                    with col2:

                                        if len(choice_img) >= 2:
                                            st.header("Pseudo Hot Coloration Image")

                                            pseudo_hot_image4 = pseudo_hot(transformed_image4, message4)
                                            apply_image4 = pseudo_hot_image4
                                            
                                        if len(choice_img) >= 5:
                                            pseudo_hot_image5 = pseudo_hot(transformed_image5, message5)
                                            apply_image5 = pseudo_hot_image5
                                            
                                        if len(choice_img) >= 8:
                                            pseudo_hot_image6 = pseudo_hot(transformed_image6, message6)
                                            apply_image6 = pseudo_hot_image6
                                        

                                    with col3:

                                        if len(choice_img) >= 3:
                                            st.header("Pseudo Hot Coloration Image")
                                        
                                            pseudo_hot_image7 = pseudo_hot(transformed_image7, message7)
                                            apply_image7 = pseudo_hot_image7
                                            
                                        if len(choice_img) >= 6:
                                            pseudo_hot_image8 = pseudo_hot(transformed_image8, message8)
                                            apply_image8 = pseudo_hot_image8
                                            
                                        if len(choice_img) >= 9:
                                            pseudo_hot_image9 = pseudo_hot(transformed_image9, message9)
                                            apply_image9 = pseudo_hot_image9
                                        
                                if choice5 == "Cool":

                                    with col1:

                                        if len(choice_img) >= 1:
                                            st.header("Pseudo Cool Coloration Image")

                                            pseudo_cool_image1 = pseudo_cool(transformed_image1, message1)
                                            apply_image1 = pseudo_cool_image1
                                            
                                        if len(choice_img) >= 4:
                                            pseudo_cool_image2 = pseudo_cool(transformed_image2, message2)
                                            apply_image2 = pseudo_cool_image2
                                            
                                        if len(choice_img) >= 7:
                                            pseudo_cool_image3 = pseudo_cool(transformed_image3, message3)
                                            apply_image3 = pseudo_cool_image3
                                        

                                    with col2:

                                        if len(choice_img) >= 2:
                                            st.header("Pseudo Cool Coloration Image")

                                            pseudo_cool_image4 = pseudo_cool(transformed_image4, message4)
                                            apply_image4 = pseudo_cool_image4
                                            
                                        if len(choice_img) >= 5:
                                            pseudo_cool_image5 = pseudo_cool(transformed_image5, message5)
                                            apply_image5 = pseudo_cool_image5
                                            
                                        if len(choice_img) >= 8:
                                            pseudo_cool_image6 = pseudo_cool(transformed_image6, message6)
                                            apply_image6 = pseudo_cool_image6
                                            

                                    with col3:

                                        if len(choice_img) >= 3:
                                            st.header("Pseudo Cool Coloration Image")
                                        
                                            pseudo_cool_image7 = pseudo_cool(transformed_image7, message7)
                                            apply_image7 = pseudo_cool_image7
                                            
                                        if len(choice_img) >= 6:
                                            pseudo_cool_image8 = pseudo_cool(transformed_image8, message8)
                                            apply_image8 = pseudo_cool_image8
                                            
                                        if len(choice_img) >= 9:
                                            pseudo_cool_image9 = pseudo_cool(transformed_image9, message9)
                                            apply_image9 = pseudo_cool_image9
                                        
                                if choice5 == "Rainbow":

                                    with col1:

                                        if len(choice_img) >= 1:
                                            st.header("Pseudo Rainbow Coloration Image")

                                            pseudo_rainbow_image1 = pseudo_rainbow(transformed_image1, message1)
                                            apply_image1 = pseudo_rainbow_image1
                                            
                                        if len(choice_img) >= 4:
                                            pseudo_rainbow_image2 = pseudo_rainbow(transformed_image2, message2)
                                            apply_image2 = pseudo_rainbow_image2
                                            
                                        if len(choice_img) >= 7:
                                            pseudo_rainbow_image3 = pseudo_rainbow(transformed_image3, message3)
                                            apply_image3 = pseudo_rainbow_image3
                                        

                                    with col2:

                                        if len(choice_img) >= 2:
                                            st.header("Pseudo Rainbow Coloration Image")

                                            pseudo_rainbow_image4 = pseudo_rainbow(transformed_image4, message4)
                                            apply_image4 = pseudo_rainbow_image4
                                            
                                        if len(choice_img) >= 5:
                                            pseudo_rainbow_image5 = pseudo_rainbow(transformed_image5, message5)
                                            apply_image5 = pseudo_rainbow_image5
                                            
                                        if len(choice_img) >= 8:
                                            pseudo_rainbow_image6 = pseudo_rainbow(transformed_image6, message6)
                                            apply_image6 = pseudo_rainbow_image6
                                        

                                    with col3:

                                        if len(choice_img) >= 3:
                                            st.header("Pseudo Rainbow Coloration Image")
                                        
                                            pseudo_rainbow_image7 = pseudo_rainbow(transformed_image7, message7)
                                            apply_image7 = pseudo_rainbow_image7
                                            
                                        if len(choice_img) >= 6:
                                            pseudo_rainbow_image8 = pseudo_rainbow(transformed_image8, message8)
                                            apply_image8 = pseudo_rainbow_image8
                                            
                                        if len(choice_img) >= 9:
                                            pseudo_rainbow_image9 = pseudo_rainbow(transformed_image9, message9)
                                            apply_image9 = pseudo_rainbow_image9
                                        
                                if choice5 == "HSV":
                                    
                                    with col1:

                                        if len(choice_img) >= 1:
                                            st.header("Pseudo HSV Coloration Image")

                                            pseudo_hsv_image1 = pseudo_hsv(transformed_image1, message1)
                                            apply_image1 = pseudo_hsv_image1
                                            
                                        if len(choice_img) >= 4:
                                            pseudo_hsv_image2 = pseudo_hsv(transformed_image2, message2)
                                            apply_image2 = pseudo_hsv_image2
                                            
                                        if len(choice_img) >= 7:
                                            pseudo_hsv_image3 = pseudo_hsv(transformed_image3, message3)
                                            apply_image3 = pseudo_hsv_image3
                                        

                                    with col2:

                                        if len(choice_img) >= 2:
                                            st.header("Pseudo HSV Coloration Image")

                                            pseudo_hsv_image4 = pseudo_hsv(transformed_image4, message4)
                                            apply_image4 = pseudo_hsv_image4
                                            
                                        if len(choice_img) >= 5:
                                            pseudo_hsv_image5 = pseudo_hsv(transformed_image5, message5)
                                            apply_image5 = pseudo_hsv_image5
                                            
                                        if len(choice_img) >= 8:
                                            pseudo_hsv_image6 = pseudo_hsv(transformed_image6, message6)
                                            apply_image6 = pseudo_hsv_image6
                                        

                                    with col3:

                                        if len(choice_img) >= 3:
                                            st.header("Pseudo HSV Coloration Image")
                                        
                                            pseudo_hsv_image7 = pseudo_hsv(transformed_image7, message7)
                                            apply_image7 = pseudo_hsv_image7
                                            
                                        if len(choice_img) >= 6:
                                            pseudo_hsv_image8 = pseudo_hsv(transformed_image8, message8)
                                            apply_image8 = pseudo_hsv_image8
                                            
                                        if len(choice_img) >= 9:
                                            pseudo_hsv_image9 = pseudo_hsv(transformed_image9, message9)
                                            apply_image9 = pseudo_hsv_image9
                                        
                                if choice5 == "JET":

                                    with col1:

                                        if len(choice_img) >= 1:
                                            st.header("Pseudo JET Coloration Image")

                                            pseudo_jet_image1 = pseudo_jet(transformed_image1, message1)
                                            apply_image1 = pseudo_jet_image1
                                            
                                        if len(choice_img) >= 4:
                                            pseudo_jet_image2 = pseudo_jet(transformed_image2, message2)
                                            apply_image2 = pseudo_jet_image2
                                            
                                        if len(choice_img) >= 7:
                                            pseudo_jet_image3 = pseudo_jet(transformed_image3, message3)
                                            apply_image3 = pseudo_jet_image3
                                        

                                    with col2:

                                        if len(choice_img) >= 2:
                                            st.header("Pseudo JET Coloration Image")

                                            pseudo_jet_image4 = pseudo_jet(transformed_image4, message4)
                                            apply_image4 = pseudo_jet_image4
                                            
                                        if len(choice_img) >= 5:
                                            pseudo_jet_image5 = pseudo_jet(transformed_image5, message5)
                                            apply_image5 = pseudo_jet_image5
                                            
                                        if len(choice_img) >= 8:
                                            pseudo_jet_image6 = pseudo_jet(transformed_image6, message6)
                                            apply_image6 = pseudo_jet_image6
                                        

                                    with col3:

                                        if len(choice_img) >= 3:
                                            st.header("Pseudo JET Coloration Image")
                                        
                                            pseudo_jet_image7 = pseudo_jet(transformed_image7, message7)
                                            apply_image7 = pseudo_jet_image7
                                            
                                        if len(choice_img) >= 6:
                                            pseudo_jet_image8 = pseudo_jet(transformed_image8, message8)
                                            apply_image8 = pseudo_jet_image8
                                            
                                        if len(choice_img) >= 9:
                                            pseudo_jet_image9 = pseudo_jet(transformed_image9, message9)
                                            apply_image9 = pseudo_jet_image9
                                            
                        if choice4 == "Pseudo Coloration" and choice5 == "Select one":
                            pass

                        else:
                            
                            if on3:
                                if choice4 == "HUE Coloration" or choice4 == "Gray Coloration":
                                    flag = 0
                                else:
                                    flag = 1
            
                                with col1:
                                    if len(choice_img) >= 1:
                                        histogram(apply_image1,flag)
                                    if len(choice_img) >= 4:
                                        histogram(apply_image2,flag)
                                    if len(choice_img) >= 7:
                                        histogram(apply_image3,flag)
                                with col2:
                                    if len(choice_img) >= 2:
                                        histogram(apply_image4,flag)
                                    if len(choice_img) >= 5:
                                        histogram(apply_image5,flag)
                                    if len(choice_img) >= 8:
                                        histogram(apply_image6,flag)
                                with col3:
                                    if len(choice_img) >= 3:
                                        histogram(apply_image7,flag)
                                    if len(choice_img) >= 6:
                                        histogram(apply_image8,flag)
                                    if len(choice_img) >= 9:
                                        histogram(apply_image9,flag)
                            
            if choice6 != "Select one":
                
                if choice4 != "Select one":

                    if choice2 != "Select one":

                        if choice2 == "No Edge Detection":
                            with st.expander("Edge Detection"):
                                # Creating of Columns                       
                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    if len(choice_img) >= 1:
                                        st.header("No Edge Detection")
                                        apply_image1 = noedge(apply_image1,message1)
                                    if len(choice_img) >= 4:
                                        apply_image2 = noedge(apply_image2,message2)
                                    if len(choice_img) >= 7:
                                        apply_image3 = noedge(apply_image3,message3)
                                with col2:
                                    if len(choice_img) >= 2:
                                        st.header("No Edge Detection")
                                        apply_image4 = noedge(apply_image4,message4)
                                    if len(choice_img) >= 5:
                                        apply_image5 = noedge(apply_image5,message5)
                                    if len(choice_img) >= 8:
                                        apply_image6 = noedge(apply_image6,message6)
                                with col3:
                                    if len(choice_img) >= 3:
                                        st.header("No Edge Detection")
                                        apply_image7 = noedge(apply_image7,message7)
                                    if len(choice_img) >= 6:
                                        apply_image8 = noedge(apply_image8,message8)
                                    if len(choice_img) >= 9:
                                        apply_image9 = noedge(apply_image9,message9)
                                    
                        else:

                            if choice3 != "Select one":
                                
                                with st.expander("Edge Detection"):
                                    # Creating of Columns                       
                                    col1, col2, col3 = st.columns([1,1,1])

                                    if choice3 == "No Filter":

                                        if len(choice_img) >= 1:
                                            image_result1 = apply_image1
                                        if len(choice_img) >= 4:
                                            image_result2 = apply_image2
                                        if len(choice_img) >= 7:
                                            image_result3 = apply_image3
                                        if len(choice_img) >= 2:
                                            image_result4 = apply_image4
                                        if len(choice_img) >= 5:
                                            image_result5 = apply_image5
                                        if len(choice_img) >= 8:
                                            image_result6 = apply_image6
                                        if len(choice_img) >= 3:
                                            image_result7 = apply_image7
                                        if len(choice_img) >= 6:
                                            image_result8 = apply_image8
                                        if len(choice_img) >= 9:
                                            image_result9 = apply_image9

                                    if choice3 == "Adaptive":
                                        # adaptive
                                        if len(choice_img) >= 1:
                                            image_result1 = cv2.adaptiveThreshold(apply_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 4:
                                            image_result2 = cv2.adaptiveThreshold(apply_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 7:
                                            image_result3 = cv2.adaptiveThreshold(apply_image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 2:
                                            image_result4 = cv2.adaptiveThreshold(apply_image4,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 5:
                                            image_result5 = cv2.adaptiveThreshold(apply_image5,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 8:
                                            image_result6 = cv2.adaptiveThreshold(apply_image6,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 3:
                                            image_result7 = cv2.adaptiveThreshold(apply_image7,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 6:
                                            image_result8 = cv2.adaptiveThreshold(apply_image8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
                                        if len(choice_img) >= 9:
                                            image_result9 = cv2.adaptiveThreshold(apply_image9,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                                    if choice3 == "Median":
                                        # median
                                        if len(choice_img) >= 1:
                                            image_result1 = cv2.medianBlur(apply_image1, 3)
                                        if len(choice_img) >= 4:
                                            image_result2 = cv2.medianBlur(apply_image2, 3)
                                        if len(choice_img) >= 7:
                                            image_result3 = cv2.medianBlur(apply_image3, 3)
                                        if len(choice_img) >= 2:
                                            image_result4 = cv2.medianBlur(apply_image4, 3)
                                        if len(choice_img) >= 5:
                                            image_result5 = cv2.medianBlur(apply_image5, 3)
                                        if len(choice_img) >= 8:
                                            image_result6 = cv2.medianBlur(apply_image6, 3)
                                        if len(choice_img) >= 3:
                                            image_result7 = cv2.medianBlur(apply_image7, 3)
                                        if len(choice_img) >= 6:
                                            image_result8 = cv2.medianBlur(apply_image8, 3)
                                        if len(choice_img) >= 9:
                                            image_result9 = cv2.medianBlur(apply_image9, 3)

                                    if choice3 == "Gaussian":
                                        # gaussian filter
                                        if len(choice_img) >= 1:
                                            image_result1 = cv2.GaussianBlur(apply_image1,(5,5),0)
                                        if len(choice_img) >= 4:
                                            image_result2 = cv2.GaussianBlur(apply_image2,(5,5),0)
                                        if len(choice_img) >= 7:
                                            image_result3 = cv2.GaussianBlur(apply_image3,(5,5),0)
                                        if len(choice_img) >= 2:
                                            image_result4 = cv2.GaussianBlur(apply_image4,(5,5),0)
                                        if len(choice_img) >= 5:
                                            image_result5 = cv2.GaussianBlur(apply_image5,(5,5),0)
                                        if len(choice_img) >= 8:
                                            image_result6 = cv2.GaussianBlur(apply_image6,(5,5),0)
                                        if len(choice_img) >= 3:
                                            image_result7 = cv2.GaussianBlur(apply_image7,(5,5),0)
                                        if len(choice_img) >= 6:
                                            image_result8 = cv2.GaussianBlur(apply_image8,(5,5),0)
                                        if len(choice_img) >= 9:
                                            image_result9 = cv2.GaussianBlur(apply_image9,(5,5),0)

                                    if choice3 == "Bilateral":
                                        # bilateral 
                                        if len(choice_img) >= 1:
                                            image_result1 = cv2.bilateralFilter(apply_image1,9,75,75)
                                        if len(choice_img) >= 4:
                                            image_result2 = cv2.bilateralFilter(apply_image2,9,75,75)
                                        if len(choice_img) >= 7:
                                            image_result3 = cv2.bilateralFilter(apply_image3,9,75,75)
                                        if len(choice_img) >= 2:
                                            image_result4 = cv2.bilateralFilter(apply_image4,9,75,75)
                                        if len(choice_img) >= 5:
                                            image_result5 = cv2.bilateralFilter(apply_image5,9,75,75)
                                        if len(choice_img) >= 8:
                                            image_result6 = cv2.bilateralFilter(apply_image6,9,75,75)
                                        if len(choice_img) >= 3:
                                            image_result7 = cv2.bilateralFilter(apply_image7,9,75,75)
                                        if len(choice_img) >= 6:
                                            image_result8 = cv2.bilateralFilter(apply_image8,9,75,75)
                                        if len(choice_img) >= 9:
                                            image_result9 = cv2.bilateralFilter(apply_image9,9,75,75)

                                    if choice3 == "Morphological":
                                        # morphological operation
                                        kernel = np.ones((5, 5), np.uint8)
                                        if len(choice_img) >= 1:
                                            image_result1 = cv2.morphologyEx(apply_image1, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 4:
                                            image_result2 = cv2.morphologyEx(apply_image2, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 7:
                                            image_result3 = cv2.morphologyEx(apply_image3, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 2:
                                            image_result4 = cv2.morphologyEx(apply_image4, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 5:
                                            image_result5 = cv2.morphologyEx(apply_image5, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 8:
                                            image_result6 = cv2.morphologyEx(apply_image6, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 3:
                                            image_result7 = cv2.morphologyEx(apply_image7, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 6:
                                            image_result8 = cv2.morphologyEx(apply_image8, cv2.MORPH_OPEN, kernel)
                                        if len(choice_img) >= 9:
                                            image_result9 = cv2.morphologyEx(apply_image9, cv2.MORPH_OPEN, kernel)

                                    if choice3 == "Averaging":
                                        # averaging filter
                                        kernel = np.ones((5,5),np.float32)/25
                                        if len(choice_img) >= 1:
                                            image_result1 = cv2.filter2D(apply_image1,-1,kernel)
                                        if len(choice_img) >= 4:
                                            image_result2 = cv2.filter2D(apply_image2,-1,kernel)
                                        if len(choice_img) >= 7:
                                            image_result3 = cv2.filter2D(apply_image3,-1,kernel)
                                        if len(choice_img) >= 2:
                                            image_result4 = cv2.filter2D(apply_image4,-1,kernel)
                                        if len(choice_img) >= 5:
                                            image_result5 = cv2.filter2D(apply_image5,-1,kernel)
                                        if len(choice_img) >= 8:
                                            image_result6 = cv2.filter2D(apply_image6,-1,kernel)
                                        if len(choice_img) >= 3:
                                            image_result7 = cv2.filter2D(apply_image7,-1,kernel)
                                        if len(choice_img) >= 6:
                                            image_result8 = cv2.filter2D(apply_image8,-1,kernel)
                                        if len(choice_img) >= 9:
                                            image_result9 = cv2.filter2D(apply_image9,-1,kernel)
                                        
                                    if choice2 == "Canny Edge Detection":
                                        with col1:
                                            if len(choice_img) >= 1:
                                                st.header("Canny Edge Detection")
                                                lower1 = st.slider("Lower Threshold Value of Image 1:",-800,800, value = 10)
                                                upper1 = st.slider("Upper Threshold Value of Image 1:",-800,800, value = 250)
                                                apply_image1 = canny(image_result1,lower1,upper1,message1)
                                            if len(choice_img) >= 4:
                                                lower2 = st.slider("Lower Threshold Value of Image 4:",-800,800, value = 10)
                                                upper2 = st.slider("Upper Threshold Value of Image 4:",-800,800, value = 250)
                                                apply_image2 = canny(image_result2,lower2,upper2,message2)
                                            if len(choice_img) >= 7:
                                                lower3 = st.slider("Lower Threshold Value of Image 7:",-800,800, value = 10)
                                                upper3 = st.slider("Upper Threshold Value of Image 7:",-800,800, value = 250)
                                                apply_image3 = canny(image_result3,lower3,upper3,message3)
                                        with col2:
                                            if len(choice_img) >= 2:
                                                st.header("Canny Edge Detection")
                                                lower4 = st.slider("Lower Threshold Value of Image 2:",-800,800, value = 10)
                                                upper4 = st.slider("Upper Threshold Value of Image 2:",-800,800, value = 250)
                                                apply_image4 = canny(image_result4,lower4,upper4,message4)
                                            if len(choice_img) >= 5:
                                                lower5 = st.slider("Lower Threshold Value of Image 5:",-800,800, value = 10)
                                                upper5 = st.slider("Upper Threshold Value of Image 5:",-800,800, value = 250)
                                                apply_image5 = canny(image_result5,lower5,upper5,message5)
                                            if len(choice_img) >= 8:
                                                lower6 = st.slider("Lower Threshold Value of Image 8:",-800,800, value = 10)
                                                upper6 = st.slider("Upper Threshold Value of Image 8:",-800,800, value = 250)
                                                apply_image6 = canny(image_result6,lower6,upper6,message6)
                                        with col3:
                                            if len(choice_img) >= 3:
                                                st.header("Canny Edge Detection")
                                                lower7 = st.slider("Lower Threshold Value of Image 3:",-800,800, value = 10)
                                                upper7 = st.slider("Upper Threshold Value of Image 3:",-800,800, value = 250)
                                                apply_image7 = canny(image_result7,lower7,upper7,message7)
                                            if len(choice_img) >= 6:
                                                lower8 = st.slider("Lower Threshold Value of Image 6:",-800,800, value = 10)
                                                upper8 = st.slider("Upper Threshold Value of Image 6:",-800,800, value = 250)
                                                apply_image8 = canny(image_result8,lower8,upper8,message8)
                                            if len(choice_img) >= 9:
                                                lower9 = st.slider("Lower Threshold Value of Image 9:",-800,800, value = 10)
                                                upper9 = st.slider("Upper Threshold Value of Image 9:",-800,800, value = 250)
                                                apply_image9 = canny(image_result9,lower9,upper9,message9)

                                    if choice2 == "Otsu Edge Detection":
                                        with col1:
                                            if len(choice_img) >= 1:
                                                st.header("Otse Edge Detection")
                                                apply_image1 = otsu(image_result1,message1)
                                            if len(choice_img) >= 4:
                                                apply_image2 = otsu(image_result2,message2)
                                            if len(choice_img) >= 7:
                                                apply_image3 = otsu(image_result3,message3)
                                        with col2:
                                            if len(choice_img) >= 2:
                                                st.header("Otse Edge Detection")
                                                apply_image4 = otsu(image_result4,message4)
                                            if len(choice_img) >= 5:
                                                apply_image5 = otsu(image_result5,message5)
                                            if len(choice_img) >= 8:
                                                apply_image6 = otsu(image_result6,message6)
                                        with col3:
                                            if len(choice_img) >= 3:
                                                st.header("Otse Edge Detection")
                                                apply_image7 = otsu(image_result7,message7)
                                            if len(choice_img) >= 6:
                                                apply_image8 = otsu(image_result8,message8)
                                            if len(choice_img) >= 9:
                                                apply_image9 = otsu(image_result9,message9)

                                    if choice2 == "Prewitt Edge Detection":
                                        with col1:
                                            if len(choice_img) >= 1:
                                                st.header("Prewitt Edge Detection")
                                                apply_image1 = prewitt(image_result1,message1)
                                            if len(choice_img) >= 4:
                                                apply_image2 = prewitt(image_result2,message2)
                                            if len(choice_img) >= 7:
                                                apply_image3 = prewitt(image_result3,message3)
                                        with col2:
                                            if len(choice_img) >= 2:
                                                st.header("Prewitt Edge Detection")
                                                apply_image4 = prewitt(image_result4,message4)
                                            if len(choice_img) >= 5:
                                                apply_image5 = prewitt(image_result5,message5)
                                            if len(choice_img) >= 8:
                                                apply_image6 = prewitt(image_result6,message6)
                                        with col3:
                                            if len(choice_img) >= 3:
                                                st.header("Prewitt Edge Detection")
                                                apply_image7 = prewitt(image_result7,message7)
                                            if len(choice_img) >= 6:
                                                apply_image8 = prewitt(image_result8,message8)
                                            if len(choice_img) >= 9:
                                                apply_image9 = prewitt(image_result9,message9)

                                    if choice2 == "Robert Edge Detection":
                                        with col1:
                                            if len(choice_img) >= 1:
                                                st.header("Robert Edge Detection")
                                                apply_image1 = robert(image_result1,message1)
                                            if len(choice_img) >= 4:
                                                apply_image2 = robert(image_result2,message2)
                                            if len(choice_img) >= 7:
                                                apply_image3 = robert(image_result3,message3)
                                        with col2:
                                            if len(choice_img) >= 2:
                                                st.header("Robert Edge Detection")
                                                apply_image4 = robert(image_result4,message4)
                                            if len(choice_img) >= 5:
                                                apply_image5 = robert(image_result5,message5)
                                            if len(choice_img) >= 8:
                                                apply_image6 = robert(image_result6,message6)
                                        with col3:
                                            if len(choice_img) >= 3:
                                                st.header("Robert Edge Detection")
                                                apply_image7 = robert(image_result7,message7)
                                            if len(choice_img) >= 6:
                                                apply_image8 = robert(image_result8,message8)
                                            if len(choice_img) >= 9:
                                                apply_image9 = robert(image_result9,message9)
                            
                            if choice2 != "No Edge Detection" and choice2 != "Select one" and choice3 == "Select one":
                                pass

                            else:
                                if on4:
                                    if choice4 == "Hue Coloration" or choice4 == "Gray Coloration" or choice2 == "Canny Edge Detection" or choice2 == "Otsu Edge Detection":
                                        flag = 0
                                    else:
                                        flag = 1

                                    with col1:
                                        if len(choice_img) >= 1:
                                            histogram(apply_image1,flag)
                                        if len(choice_img) >= 4:
                                            histogram(apply_image2,flag)
                                        if len(choice_img) >= 7:
                                            histogram(apply_image3,flag)
                                    with col2:
                                        if len(choice_img) >= 2:
                                            histogram(apply_image4,flag)
                                        if len(choice_img) >= 5:
                                            histogram(apply_image5,flag)
                                        if len(choice_img) >= 8:
                                            histogram(apply_image6,flag)
                                    with col3:
                                        if len(choice_img) >= 3:
                                            histogram(apply_image7,flag)
                                        if len(choice_img) >= 6:
                                            histogram(apply_image8,flag)
                                        if len(choice_img) >= 9:
                                            histogram(apply_image9,flag)

        with tab3:
            if choice7 != "Select one":
                if choice6 != "Select one":
                    
                    if choice4 != "Select one":

                        if choice2 != "Select one":

                            if choice2 != "No Edge Detection" and choice3 == "Select one":
                                pass

                            else:
                                    
                                if on6:

                                    # with st.expander("Mathematical Matrics"):
                                        # Creating of Columns                       
                                        # col1, col2, col3 = st.columns([1,1,1])

                                        if on7:
                                            flag = 1
                                        else:
                                            flag = 0

                                        with st.expander("Average Graph"):
                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                if flag == 1:
                                                    st.subheader("Average Line Chart")
                                                if len(choice_img) >= 1:
                                                    avg_total = mean(apply_image1,flag,message1)
                                                    avg_total1 = avg_total
                                                if len(choice_img) >= 4:
                                                    avg_total = mean(apply_image2,flag,message2)
                                                    avg_total2 = avg_total
                                                if len(choice_img) >= 7:
                                                    avg_total = mean(apply_image3,flag,message3)
                                                    avg_total3 = avg_total

                                            with col2:
                                                if flag == 1:
                                                    st.subheader("Average Line Chart")
                                                if len(choice_img) >= 2:
                                                    avg_total = mean(apply_image4,flag,message4)
                                                    avg_total4 = avg_total
                                                if len(choice_img) >= 5:
                                                    avg_total = mean(apply_image5,flag,message5)
                                                    avg_total5 = avg_total
                                                if len(choice_img) >= 8:
                                                    avg_total = mean(apply_image6,flag,message6)
                                                    avg_total6 = avg_total

                                            with col3:
                                                if flag == 1:
                                                    st.subheader("Average Line Chart")
                                                if len(choice_img) >= 3:
                                                    avg_total = mean(apply_image7,flag,message7)
                                                    avg_total7 = avg_total
                                                if len(choice_img) >= 6:
                                                    avg_total = mean(apply_image8,flag,message8)
                                                    avg_total8 = avg_total
                                                if len(choice_img) >= 9:
                                                    avg_total = mean(apply_image9,flag,message9)
                                                    avg_total9 = avg_total

                                        with st.expander("Standard Deviation Graph"):
                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                if flag == 1:
                                                    st.subheader("Standard Deviation Line Chart")
                                                if len(choice_img) >= 1:
                                                    std_total = std(apply_image1,flag,message1)
                                                    std_total1 = std_total
                                                if len(choice_img) >= 4:
                                                    std_total = std(apply_image2,flag,message2)
                                                    std_total2 = std_total
                                                if len(choice_img) >= 7:
                                                    std_total = std(apply_image3,flag,message3)
                                                    std_total3 = std_total

                                            with col2:
                                                if flag == 1:
                                                    st.subheader("Standard Deviation Line Chart")
                                                if len(choice_img) >= 2:
                                                    std_total = std(apply_image4,flag,message4)
                                                    std_total4 = std_total
                                                if len(choice_img) >= 5:
                                                    std_total = std(apply_image5,flag,message5)
                                                    std_total5 = std_total
                                                if len(choice_img) >= 8:
                                                    std_total = std(apply_image6,flag,message6)
                                                    std_total6 = std_total

                                            with col3:
                                                if flag == 1:
                                                    st.subheader("Standard Deviation Line Chart")
                                                if len(choice_img) >= 3:
                                                    std_total = std(apply_image7,flag,message7)
                                                    std_total7 = std_total
                                                if len(choice_img) >= 6:
                                                    std_total = std(apply_image8,flag,message8)
                                                    std_total8 = std_total
                                                if len(choice_img) >= 9:
                                                    std_total = std(apply_image9,flag,message9)
                                                    std_total9 = std_total

                                        with st.expander("Variance Graph"):
                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                if flag == 1:
                                                    st.subheader("Variation Line Chart")
                                                if len(choice_img) >= 1:
                                                    var_total = var(apply_image1,flag,message1)
                                                    var_total1 = var_total
                                                if len(choice_img) >= 4:
                                                    var_total = var(apply_image2,flag,message2)
                                                    var_total2 = var_total
                                                if len(choice_img) >= 7:
                                                    var_total = var(apply_image3,flag,message3)
                                                    var_total3 = var_total
                                            
                                            with col2:
                                                if flag == 1:
                                                    st.subheader("Variation Line Chart")
                                                if len(choice_img) >= 2:
                                                    var_total = var(apply_image4,flag,message4)
                                                    var_total4 = var_total
                                                if len(choice_img) >= 5:
                                                    var_total = var(apply_image5,flag,message5)
                                                    var_total5 = var_total
                                                if len(choice_img) >= 8:
                                                    var_total = var(apply_image6,flag,message6)
                                                    var_total6 = var_total
                                                
                                            with col3:
                                                if flag == 1:
                                                    st.subheader("Variation Line Chart")
                                                if len(choice_img) >= 3:
                                                    var_total = var(apply_image7,flag,message7)
                                                    var_total7 = var_total
                                                if len(choice_img) >= 6:
                                                    var_total = var(apply_image8,flag,message8)
                                                    var_total8 = var_total
                                                if len(choice_img) >= 9:
                                                    var_total = var(apply_image9,flag,message9)
                                                    var_total9 = var_total

                                        with st.expander("RMS Graph"):
                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                if flag == 1:
                                                    st.subheader("Root Mean Square Line Chart")
                                                if len(choice_img) >= 1:
                                                    rms_total = rms(apply_image1,flag,message1)
                                                    rms_total1 = rms_total
                                                if len(choice_img) >= 4:
                                                    rms_total = rms(apply_image2,flag,message2)
                                                    rms_total2 = rms_total
                                                if len(choice_img) >= 7:
                                                    rms_total = rms(apply_image3,flag,message3)
                                                    rms_total3 = rms_total

                                            with col2:
                                                if flag == 1:
                                                    st.subheader("Root Mean Square Line Chart")
                                                if len(choice_img) >= 2:
                                                    rms_total = rms(apply_image4,flag,message4)
                                                    rms_total4 = rms_total
                                                if len(choice_img) >= 5:
                                                    rms_total = rms(apply_image5,flag,message5)
                                                    rms_total5 = rms_total
                                                if len(choice_img) >= 8:
                                                    rms_total = rms(apply_image6,flag,message6)
                                                    rms_total6 = rms_total

                                            with col3:
                                                if flag == 1:
                                                    st.subheader("Root Mean Square Line Chart")
                                                if len(choice_img) >= 3:
                                                    rms_total = rms(apply_image7,flag,message7)
                                                    rms_total7 = rms_total
                                                if len(choice_img) >= 6:
                                                    rms_total = rms(apply_image8,flag,message8)
                                                    rms_total8 = rms_total
                                                if len(choice_img) >= 9:
                                                    rms_total = rms(apply_image9,flag,message9)
                                                    rms_total9 = rms_total

                                        with tab2:
                                            with st.expander("Summary Table"):
                                                col1, col2, col3 = st.columns([1,1,1])
                                                with col1:
                                                    if len(choice_img) >= 1:
                                                        skew_total = skew(apply_image1)
                                                        skew_total1 = skew_total
                                                    if len(choice_img) >= 4:
                                                        skew_total = skew(apply_image2)
                                                        skew_total2 = skew_total
                                                    if len(choice_img) >= 7:
                                                        skew_total = skew(apply_image3)
                                                        skew_total3 = skew_total
                                                        
                                                    if len(choice_img) >= 1:
                                                        kur_total = kur(apply_image1)
                                                        kur_total1 = kur_total
                                                    if len(choice_img) >= 4:
                                                        kur_total = kur(apply_image2)
                                                        kur_total2 = kur_total
                                                    if len(choice_img) >= 7:
                                                        kur_total = kur(apply_image3)
                                                        kur_total3 = kur_total

                                                    if len(choice_img) >= 1:
                                                        entropy_total = entropy(apply_image1)
                                                        entropy_total1 = entropy_total
                                                    if len(choice_img) >= 4:
                                                        entropy_total = entropy(apply_image2)
                                                        entropy_total2 = entropy_total
                                                    if len(choice_img) >= 7:
                                                        entropy_total = entropy(apply_image3)
                                                        entropy_total3 = entropy_total

                                                    if len(choice_img) >= 1:
                                                        hist_total = histogram_mean(apply_image1)
                                                        hist_total1 = hist_total
                                                    if len(choice_img) >= 4:
                                                        hist_total = histogram_mean(apply_image2)
                                                        hist_total2 = hist_total
                                                    if len(choice_img) >= 7:
                                                        hist_total = histogram_mean(apply_image3)
                                                        hist_total3 = hist_total

                                                with col2:

                                                    if len(choice_img) >= 2:
                                                        skew_total = skew(apply_image4)
                                                        skew_total4 = skew_total
                                                    if len(choice_img) >= 5:
                                                        skew_total = skew(apply_image5)
                                                        skew_total5 = skew_total
                                                    if len(choice_img) >= 8:
                                                        skew_total = skew(apply_image6)
                                                        skew_total6 = skew_total

                                                    if len(choice_img) >= 2:
                                                        kur_total = kur(apply_image4)
                                                        kur_total4 = kur_total
                                                    if len(choice_img) >= 5:
                                                        kur_total = kur(apply_image5)
                                                        kur_total5 = kur_total
                                                    if len(choice_img) >= 8:
                                                        kur_total = kur(apply_image6)
                                                        kur_total6 = kur_total

                                                    if len(choice_img) >= 2:
                                                        entropy_total = entropy(apply_image4)
                                                        entropy_total4 = entropy_total
                                                    if len(choice_img) >= 5:
                                                        entropy_total = entropy(apply_image5)
                                                        entropy_total5 = entropy_total
                                                    if len(choice_img) >= 8:
                                                        entropy_total = entropy(apply_image6)
                                                        entropy_total6 = entropy_total

                                                    if len(choice_img) >= 2:
                                                        hist_total = histogram_mean(apply_image4)
                                                        hist_total4 = hist_total
                                                    if len(choice_img) >= 5:
                                                        hist_total = histogram_mean(apply_image5)
                                                        hist_total5 = hist_total
                                                    if len(choice_img) >= 8:
                                                        hist_total = histogram_mean(apply_image6)
                                                        hist_total6 = hist_total

                                                with col3:

                                                    if len(choice_img) >= 3:
                                                        skew_total = skew(apply_image7)
                                                        skew_total7 = skew_total
                                                    if len(choice_img) >= 6:
                                                        skew_total = skew(apply_image8)
                                                        skew_total8 = skew_total
                                                    if len(choice_img) >= 9:
                                                        skew_total = skew(apply_image9)
                                                        skew_total9 = skew_total
                                                        
                                                    if len(choice_img) >= 3:
                                                        kur_total = kur(apply_image7)
                                                        kur_total7 = kur_total
                                                    if len(choice_img) >= 6:
                                                        kur_total = kur(apply_image8)
                                                        kur_total8 = kur_total
                                                    if len(choice_img) >= 9:
                                                        kur_total = kur(apply_image9)
                                                        kur_total9 = kur_total

                                                    if len(choice_img) >= 3:
                                                        entropy_total = entropy(apply_image7)
                                                        entropy_total7 = entropy_total
                                                    if len(choice_img) >= 6:
                                                        entropy_total = entropy(apply_image8)
                                                        entropy_total8 = entropy_total
                                                    if len(choice_img) >= 9:
                                                        entropy_total = entropy(apply_image9)
                                                        entropy_total9 = entropy_total

                                                    if len(choice_img) >= 3:
                                                        hist_total = histogram_mean(apply_image7)
                                                        hist_total7 = hist_total
                                                    if len(choice_img) >= 6:
                                                        hist_total = histogram_mean(apply_image8)
                                                        hist_total8 = hist_total
                                                    if len(choice_img) >= 9:
                                                        hist_total = histogram_mean(apply_image9)
                                                        hist_total9 = hist_total

                                                if choice2 == "No Edge Detection" or (choice2 != "No Edge Detection" and choice3 != "Select one"):

                                                        if on6:

                                                            if len(choice_img) == 9:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7,message2,message5,message8,message3,message6,message9]
                                                                table_mean = [avg_total1,avg_total4,avg_total7,avg_total2,avg_total5,avg_total8,avg_total3,avg_total6,avg_total9]
                                                                table_std = [std_total1,std_total4,std_total7,std_total2,std_total5,std_total8,std_total3,std_total6,std_total9]
                                                                table_var = [var_total1,var_total4,var_total7,var_total2,var_total5,var_total8,var_total3,var_total6,var_total9]
                                                                table_rms = [rms_total1,rms_total4,rms_total7,rms_total2,rms_total5,rms_total8,rms_total3,rms_total6,rms_total9]
                                                                table_skew = [skew_total1,skew_total4,skew_total7,skew_total2,skew_total5,skew_total8,skew_total3,skew_total6,skew_total9]
                                                                table_kur = [kur_total1,kur_total4,kur_total7,kur_total2,kur_total5,kur_total8,kur_total3,kur_total6,kur_total9]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7,entropy_total2,entropy_total5,entropy_total8,entropy_total3,entropy_total6,entropy_total9]
                                                                table_hist = [hist_total1,hist_total4,hist_total7,hist_total2,hist_total5,hist_total8,hist_total3,hist_total6,hist_total9]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 8:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7,message2,message5,message8,message3,message6]
                                                                table_mean = [avg_total1,avg_total4,avg_total7,avg_total2,avg_total5,avg_total8,avg_total3,avg_total6]
                                                                table_std = [std_total1,std_total4,std_total7,std_total2,std_total5,std_total8,std_total3,std_total6]
                                                                table_var = [var_total1,var_total4,var_total7,var_total2,var_total5,var_total8,var_total3,var_total6]
                                                                table_rms = [rms_total1,rms_total4,rms_total7,rms_total2,rms_total5,rms_total8,rms_total3,rms_total6]
                                                                table_skew = [skew_total1,skew_total4,skew_total7,skew_total2,skew_total5,skew_total8,skew_total3,skew_total6]
                                                                table_kur = [kur_total1,kur_total4,kur_total7,kur_total2,kur_total5,kur_total8,kur_total3,kur_total6]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7,entropy_total2,entropy_total5,entropy_total8,entropy_total3,entropy_total6]
                                                                table_hist = [hist_total1,hist_total4,hist_total7,hist_total2,hist_total5,hist_total8,hist_total3,hist_total6]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 7:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7,message2,message5,message8,message3]
                                                                table_mean = [avg_total1,avg_total4,avg_total7,avg_total2,avg_total5,avg_total8,avg_total3]
                                                                table_std = [std_total1,std_total4,std_total7,std_total2,std_total5,std_total8,std_total3]
                                                                table_var = [var_total1,var_total4,var_total7,var_total2,var_total5,var_total8,var_total3]
                                                                table_rms = [rms_total1,rms_total4,rms_total7,rms_total2,rms_total5,rms_total8,rms_total3]
                                                                table_skew = [skew_total1,skew_total4,skew_total7,skew_total2,skew_total5,skew_total8,skew_total3]
                                                                table_kur = [kur_total1,kur_total4,kur_total7,kur_total2,kur_total5,kur_total8,kur_total3]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7,entropy_total2,entropy_total5,entropy_total8,entropy_total3]
                                                                table_hist = [hist_total1,hist_total4,hist_total7,hist_total2,hist_total5,hist_total8,hist_total3]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 6:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7,message2,message5,message8]
                                                                table_mean = [avg_total1,avg_total4,avg_total7,avg_total2,avg_total5,avg_total8]
                                                                table_std = [std_total1,std_total4,std_total7,std_total2,std_total5,std_total8]
                                                                table_var = [var_total1,var_total4,var_total7,var_total2,var_total5,var_total8]
                                                                table_rms = [rms_total1,rms_total4,rms_total7,rms_total2,rms_total5,rms_total8]
                                                                table_skew = [skew_total1,skew_total4,skew_total7,skew_total2,skew_total5,skew_total8]
                                                                table_kur = [kur_total1,kur_total4,kur_total7,kur_total2,kur_total5,kur_total8]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7,entropy_total2,entropy_total5,entropy_total8]
                                                                table_hist = [hist_total1,hist_total4,hist_total7,hist_total2,hist_total5,hist_total8]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 5:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7,message2,message5]
                                                                table_mean = [avg_total1,avg_total4,avg_total7,avg_total2,avg_total5]
                                                                table_std = [std_total1,std_total4,std_total7,std_total2,std_total5]
                                                                table_var = [var_total1,var_total4,var_total7,var_total2,var_total5]
                                                                table_rms = [rms_total1,rms_total4,rms_total7,rms_total2,rms_total5]
                                                                table_skew = [skew_total1,skew_total4,skew_total7,skew_total2,skew_total5]
                                                                table_kur = [kur_total1,kur_total4,kur_total7,kur_total2,kur_total5]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7,entropy_total2,entropy_total5]
                                                                table_hist = [hist_total1,hist_total4,hist_total7,hist_total2,hist_total5]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 4:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7,message2]
                                                                table_mean = [avg_total1,avg_total4,avg_total7,avg_total2]
                                                                table_std = [std_total1,std_total4,std_total7,std_total2]
                                                                table_var = [var_total1,var_total4,var_total7,var_total2]
                                                                table_rms = [rms_total1,rms_total4,rms_total7,rms_total2]
                                                                table_skew = [skew_total1,skew_total4,skew_total7,skew_total2]
                                                                table_kur = [kur_total1,kur_total4,kur_total7,kur_total2]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7,entropy_total2]
                                                                table_hist = [hist_total1,hist_total4,hist_total7,hist_total2]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 3:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4,message7]
                                                                table_mean = [avg_total1,avg_total4,avg_total7]
                                                                table_std = [std_total1,std_total4,std_total7]
                                                                table_var = [var_total1,var_total4,var_total7]
                                                                table_rms = [rms_total1,rms_total4,rms_total7]
                                                                table_skew = [skew_total1,skew_total4,skew_total7]
                                                                table_kur = [kur_total1,kur_total4,kur_total7]
                                                                table_entropy = [entropy_total1,entropy_total4,entropy_total7]
                                                                table_hist = [hist_total1,hist_total4,hist_total7]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)
                                                            
                                                            elif len(choice_img) == 2:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1,message4]
                                                                table_mean = [avg_total1,avg_total4]
                                                                table_std = [std_total1,std_total4]
                                                                table_var = [var_total1,var_total4]
                                                                table_rms = [rms_total1,rms_total4]
                                                                table_skew = [skew_total1,skew_total4]
                                                                table_kur = [kur_total1,kur_total4]
                                                                table_entropy = [entropy_total1,entropy_total4]
                                                                table_hist = [hist_total1,hist_total4]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)

                                                            elif len(choice_img) == 1:
                                                                st.subheader("Summary Table")
                                                                table_row = [message1]
                                                                table_mean = [avg_total1]
                                                                table_std = [std_total1]
                                                                table_var = [var_total1]
                                                                table_rms = [rms_total1]
                                                                table_skew = [skew_total1]
                                                                table_kur = [kur_total1]
                                                                table_entropy = [entropy_total1]
                                                                table_hist = [hist_total1]
                                                                
                                                                # dictionary of lists 
                                                                dict = {"Category": table_row, "Average": table_mean, "Standard Deviation": table_std, "Variance": table_var, "Root Mean Square": table_rms, "Skewness": table_skew, "Kurtosis": table_kur, "Entropy": table_entropy, "Histogram Mean": table_hist}
                                                                df = pd.DataFrame(dict)
                                                                st.dataframe(df)
