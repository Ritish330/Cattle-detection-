import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_one_box(xyxy, im, label="", color=None, line_thickness=None):
    # Extract coordinates
    x_min, y_min, x_max, y_max = xyxy

    # Adjust coordinates to stay within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(im.shape[1], x_max)
    y_max = min(im.shape[0], y_max)

    # Draw bounding box
    color = colors(None, True) if color is None else color
    line_thickness = 3 if line_thickness is None else line_thickness
    im = cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, line_thickness)

    # Draw filled rectangle as background for text
    tf = max(line_thickness - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
    im = cv2.rectangle(
        im,
        (x_min + 2, y_min),
        (x_min + t_size[0] + 2, y_min + t_size[1] + 2),
        color,
        cv2.FILLED,
    )

    # Draw label
    im = cv2.putText(
        im,
        label,
        (x_min + 2, y_min + t_size[1] + 2),
        0,
        line_thickness / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )

    return im
def colors(c, bgr=True):
    """
    Define a color scheme for bounding boxes based on class index.
    Args:
        c (int): Class index.
        bgr (bool): If True, returns BGR, else RGB.

    Returns:
        List[int]: RGB color.
    """
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 255),  # White
        (0, 0, 0),  # Black
    ]
    return colors[c % len(colors)] if bgr else [c[::-1] for c in colors[c % len(colors)]]

def plot_graph(data):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(data)), data.values(), align='center')
    plt.xticks(range(len(data)), list(data.keys()), rotation=45)
    plt.xlabel('Cattle Classes')
    plt.ylabel('Count')
    plt.title('Cattle Detection Results')
    return plt

def update_graph(ax, data):
    ax.clear()
    ax.bar(range(len(data)), data.values(), align='center')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(list(data.keys()), rotation=45)
    ax.set_xlabel('Cattle Classes')
    ax.set_ylabel('Count')
    ax.set_title('Cattle Detection Results')
    return ax

def detect(file_path, label, count_label, health_label, ax):
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path="weights/yolov5s.pt")
    except:
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", source="local", path="weights/yolov5s.pt"
        )

    model.conf = 0.4  # Set confidence threshold 
    model.iou = 0.4  # Set IOU threshold 

    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names

    # Filter indices for the specified Cattles
    target_animals = ["hen", "duck", "cow", "sheep", "goat", "donkey", "ox", "bull", "horse", "bird"]
    target_indices = [class_index for class_index, class_name in names.items() if class_name in target_animals]

    # Perform object detection
    image = cv2.imread(file_path)  # Read the selected image with OpenCV
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert image to RGB (OpenCV uses BGR by default)
    results = model(image)  # Get detection results

    # Process detections
    count = 0
    detection_data = {class_name: 0 for class_name in target_animals}
    healthy_animals = 0

    for det in results.xyxy[0]:  # detections per image
        xyxy = det[:4].cpu().numpy().astype(int)
        conf = det[4].cpu().numpy()
        class_index = int(det[5].cpu().numpy())

        # Check if the detected class is one of the specified Cattles
        if class_index in target_indices and conf >= model.conf:
            label_text = names[class_index] + f" {conf:.2f}"
            
            # Draw bounding box and label on the image
            image = plot_one_box(xyxy, image, label=label_text, color=colors(class_index, True))

            # Increment count for each detection
            count += 1
            detection_data[names[class_index]] += 1

            # Simulate guessing animal health
            is_healthy = random.choice([True, False])  # Randomly guessing health
            if is_healthy:
                healthy_animals += 1

    # Update the count label with the number of detections or "No Cattles Detected" message
    if count > 0:
        count_label.config(text=f"Number of Cattles: {count}", font=("Arial", 18))
        health_label.config(
            text=f"Healthy Cattles: {healthy_animals}", font=("Arial", 18)
        )

        # Update the graph
        update_graph(ax, detection_data)
        canvas.draw()

    else:
        count_label.config(text="No Cattles Detected", font=("Arial", 18))
        health_label.config(text="No health assessment available", font=("Arial", 18))

    # Convert the result to PIL Image
    result_image = Image.fromarray(image)
    result_image = result_image.resize((400, 400))  # Resize image to 400x400

    # Add a border to the image
    result_image_with_border = Image.new(
        "RGB", (result_image.width + 10, result_image.height + 10), color="black"
    )
    result_image_with_border.paste(result_image, (5, 5))

    result_image_with_border = ImageTk.PhotoImage(result_image_with_border)

    # Update the label in the Tkinter window
    label.config(image=result_image_with_border)
    label.image = result_image_with_border
    count_label.update()
    health_label.update()

# Create a Tkinter window with specified colors
root = tk.Tk()
root.title("Cattle Detection System")
root.config(bg="gray")  # Set background color to #7e22ce

# Center the window on the screen
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

file_path = None

# Create a label to display the resulting image
label = tk.Label(root, borderwidth=2, relief="solid")
label.pack()

# Create a label to display the count of Cattles or message when no Cattles detected
count_label = tk.Label(
    root, text="Number of Cattles: 0", bg="#7e22ce", fg="white", font=("Arial", 18)
)
count_label.pack()

# Create a label to display the health of Cattles
health_label = tk.Label(
    root, text="Healthy Cattles: 0", bg="#7e22ce", fg="white", font=("Arial", 18)
)
health_label.pack()

# Create a Figure and Axes for Matplotlib
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Function to perform object detection on the selected image
def detect_objects():
    global file_path
    file_path = filedialog.askopenfilename()  # Open a file dialog to choose the image
    if file_path:
        # Align the button at the bottom after selecting the image
        btn.pack_forget()
        btn.pack(side=tk.BOTTOM)

        # Perform object detection and update the Tkinter window
        detect(file_path, label, count_label, health_label, ax)

# Create a button to select and process the image
btn = tk.Button(
    root,
    text="Select Image",
    command=detect_objects,
    bg="white",
    fg="black",
    width=40,
    padx=10,
    pady=5,
)  # Set button size, padding, and colors
btn.pack(side=tk.BOTTOM)  # Position button at the bottom

# Run the Tkinter main loop
root.mainloop()
