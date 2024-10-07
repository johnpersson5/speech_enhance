import os
os.environ["PYTORCH_JIT"] = "0"
import torch
import torchaudio
import gc
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from pydub import AudioSegment
from threading import Thread
from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def clear_gpu_cash():
    # del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def convert_to_wav(input_path):
    audio = AudioSegment.from_file(input_path)
    temp_wav_path = "temp_audio.wav"
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path
def _fn(path, solver, nfe, tau,chunk_seconds,chunks_overlap, denoising):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1
    # Convert the audio file to WAV format
    temp_wav_path = convert_to_wav(file_path)
    dwav, sr = torchaudio.load(temp_wav_path)

    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav = dwav, sr = sr, device = device, nfe=nfe,chunk_seconds=chunk_seconds,chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    clear_gpu_cash()
    return (new_sr, wav1), (new_sr, wav2)


# def main():
#     inputs: list = [
#         gr.Audio(type="filepath", label="Input Audio"),
#         gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver (Midpoint is recommended)"),
#         gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)"),
#         gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature (higher values can improve quality but can reduce stability)"),
#         gr.Slider(minimum=1, maximum=40, value=10, step=1, label="Chunk seconds (more secods more VRAM usage)"),
#         gr.Slider(minimum=0, maximum=5, value=1, step=0.5, label="Chunk overlap"),
#         # chunk_seconds, chunks_overlap
#         gr.Checkbox(value=False, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)"),
#     ]
#
#     outputs: list = [
#         gr.Audio(label="Output Denoised Audio"),
#         gr.Audio(label="Output Enhanced Audio"),
#     ]
#
#     interface = gr.Interface(
#         fn=_fn,
#         title="Resemble Enhance",
#         description="AI-driven audio enhancement for your audio files, powered by Resemble AI.",
#         inputs=inputs,
#         outputs=outputs,
#     )
#
#     interface.launch()
#
#
# if __name__ == "__main__":
#     main()
def submit():
    if not file_path:
        tk.messagebox.showerror("Error", "Please select an audio file before submitting.")
        return
    progress_bar.start()
    # Get values from sliders and checkbox
    chunk_seconds = chunk_seconds_slider.get()
    chunks_overlap = chunk_overlap_slider.get()
    nfe = number_of_functions_slider.get()
    tau = temperature_slider.get()

    denoising = denoise_var.get()

    # Default parameters (you can adjust these as needed)
    solver = 'Midpoint'  # Example solver, we could add a drop down for RK4 and Euler if we want
    #nfe = 64        # Number of function evaluations
    #tau = 0.5       # Tau parameter

    try:
        # Call the processing function
        result1, result2 = _fn(
            path=file_path,
            solver=solver,
            nfe=nfe,
            tau=tau,
            chunk_seconds=chunk_seconds,
            chunks_overlap=chunks_overlap,
            denoising=denoising
        )

        if result1 is None and result2 is None:
            tk.messagebox.showerror("Error", "An error occurred during processing.")
            return

        # Save the resulting sound files
        original_dir = os.path.dirname(file_path)
        original_filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(original_filename)

        sr1, wav1 = result1
        sr2, wav2 = result2

        # Convert numpy arrays back to tensors for saving
        wav1_tensor = torch.tensor(wav1).unsqueeze(0)
        wav2_tensor = torch.tensor(wav2).unsqueeze(0)
        # Save denoised file
        #denoise_filename = os.path.join(original_dir, f"{base_name}_denoise{ext}")
        denoise_filename = os.path.join(original_dir, f"{base_name}_denoise.wav")
        torchaudio.save(denoise_filename, wav1_tensor, sr1)
        # Save enhanced file
        #enhance_filename = os.path.join(original_dir, f"{base_name}_enhance{ext}")
        enhance_filename = os.path.join(original_dir, f"{base_name}_enhance.wav")
        torchaudio.save(enhance_filename, wav2_tensor, sr2)

        tk.messagebox.showinfo(
            "Success",
            f"Files saved successfully:\n{denoise_filename}\n{enhance_filename}"
        )
        progress_bar.stop()
    except Exception as e:
        tk.messagebox.showerror("Processing Error", f"An error occurred:\n{e}")

def threading_submit():
    t1=Thread(target=submit)
    t1.start()
# Initialize global variables
file_path = None

def upload_file():
    global file_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.aac;*.m4a")]
    )
    if file_path:
        file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

# Create the main application window
root = tk.Tk()
root.title("Podcast Audio Enhancer")

# Set window dimensions (e.g., 600x400 pixels)
window_width = 800
window_height = 600

# Get the screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position to center the window
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Set the geometry of the window (width x height + X position + Y position)
root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

# Top label
title_label = tk.Label(root, text="Podcast Audio Enhancer", font=("Arial", 16))
title_label.pack(pady=10)

# File upload button and label
upload_button = tk.Button(root, text="Select Audio File", command=upload_file)

upload_button.pack(pady=5)

file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=5)

# "Chunk Seconds" slider
chunk_seconds_label = tk.Label(root, text="Chunk Seconds")
chunk_seconds_label.pack()
chunk_seconds_slider = tk.Scale(
    root, from_=1, to=40, orient=tk.HORIZONTAL, length=400
)
chunk_seconds_slider.set(10)  # Set default value to 10
chunk_seconds_slider.pack(pady=5)

# "Chunk Overlap" slider
chunk_overlap_label = tk.Label(root, text="Chunk Overlap")
chunk_overlap_label.pack()
chunk_overlap_slider = tk.Scale(
    root, from_=0, to=5, orient=tk.HORIZONTAL, length=400
)
chunk_overlap_slider.set(1)  # Set default value to 1
chunk_overlap_slider.pack(pady=5)

# "Number of Function Evaluations" slider
number_of_functions_label = tk.Label(root, text="Number of Function Evaluations")
number_of_functions_label.pack()
number_of_functions_slider = tk.Scale(
    root, from_=1, to=128, orient=tk.HORIZONTAL, length=400
)
number_of_functions_slider.set(128)  # Set default value to 1
number_of_functions_slider.pack(pady=5)

# "Temperature" slider
temperature_label = tk.Label(root, text="Temperature")
temperature_label.pack()
temperature_slider = tk.Scale(
    root, from_=0, to=1, orient=tk.HORIZONTAL, length=400, resolution=0.01
)
temperature_slider.set(0.4)  # Set default value to 1
temperature_slider.pack(pady=5)

# "Denoise Before Enhancement" checkbox
denoise_var = tk.BooleanVar()
denoise_checkbox = tk.Checkbutton(
    root, text="Denoise Before Enhancement", variable=denoise_var
)
denoise_checkbox.pack(pady=5)

# Submit button
#submit_button = tk.Button(root, text="Submit", command=submit)
submit_button = tk.Button(root, text="Submit", command=threading_submit)

submit_button.pack(pady=20)

#Progress Bar
progress_bar = ttk.Progressbar(root,mode='indeterminate',length=300)
progress_bar.pack(pady=20)

# Run the application
root.mainloop()