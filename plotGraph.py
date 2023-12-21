import tkinter as tk
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_weights_histogram(model):
    # Create a histogram of the weights
    fig, ax = plt.subplots(figsize=(18, 10))
    
    colors = ['blue', 'green', 'orange', 'red']  # Add more colors if needed
    bins = 50
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            ax.hist(param.data.cpu().numpy().flatten(), bins=bins, alpha=0.5, color=colors[i % len(colors)])

    ax.grid(True)

    return fig

def select_and_plot():
    # Load the YOLOv5s model with automatic download
    yolov5s_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Plot the weights histogram
    fig = plot_weights_histogram(yolov5s_model)

    # Embed the Matplotlib figure in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Keep the window alive
    root.mainloop()

# Create a Tkinter window
root = tk.Tk()
root.title("YOLOv5s Weights Histogram Plotter")
root.geometry("900x600")

# Create a button to select and plot the weights histogram
btn = tk.Button(
    root,
    text="Plot YOLOv5s Weights Histogram",
    command=select_and_plot,
    bg="white",
    fg="black",
    width=40,
    padx=10,
    pady=5,
)
btn.pack(pady=20)

root.mainloop()
