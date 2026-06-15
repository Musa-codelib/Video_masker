import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os, sys, traceback, time
from engine import run_masker_engine

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Masker Pro")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        # 1. Start-up Health Check
        self.check_system()

        # 2. Branding (Logo)
        try:
            logo_path = get_resource_path("logo.png")
            img = Image.open(logo_path)
            img = img.resize((150, 150))
            self.logo_img = ImageTk.PhotoImage(img)
            tk.Label(root, image=self.logo_img).pack(pady=20)
        except:
            tk.Label(root, text="[ Logo Missing ]", fg="red").pack(pady=20)

        tk.Label(root, text="AI Masker Pro", font=("Arial", 28, "bold")).pack()
        tk.Label(root, text="v1.0 - Apple Silicon Optimized", font=("Arial", 10), fg="gray").pack()

        # 3. Path Selections
        self.input_file = ""
        self.output_dir = ""
        self.mode = tk.StringVar(value="prores")

        tk.Button(root, text="Step 1: Select Input Video", width=30, command=self.set_in).pack(pady=15)
        self.lbl_in = tk.Label(root, text="No video selected", wraplength=450, fg="blue")
        self.lbl_in.pack()

        tk.Button(root, text="Step 2: Select Output Folder", width=30, command=self.set_out).pack(pady=10)
        self.lbl_out = tk.Label(root, text="No folder selected", wraplength=450, fg="blue")
        self.lbl_out.pack()

        # 4. Mode Selection
        m_frame = tk.LabelFrame(root, text="Step 3: Export Mode", padx=20, pady=10)
        m_frame.pack(pady=20)
        tk.Radiobutton(m_frame, text="ProRes 4444 (Alpha)", variable=self.mode, value="prores").pack(side="left", padx=10)
        tk.Radiobutton(m_frame, text="B&W Mask (Fusion)", variable=self.mode, value="bw").pack(side="left", padx=10)

        # 5. Launch Button
        self.btn = tk.Button(root, text="LAUNCH AI SELECTOR", bg="#2ecc71", fg="black", 
                            font=("Arial", 16, "bold"), height=2, width=25, command=self.start)
        self.btn.pack(pady=10)

    def check_system(self):
        """ The Requirement Wizard: Check for critical bundle assets """
        missing = []
        if not os.path.exists(get_resource_path("ffmpeg")): missing.append("FFmpeg Engine")
        if not os.path.exists(get_resource_path("checkpoints/sam2_hiera_small.pt")): missing.append("AI Weights (.pt)")
        if not os.path.exists(get_resource_path("sam2_hiera_s.yaml")): missing.append("AI Config (.yaml)")
        
        if missing:
            messagebox.showerror("Requirement Error", "The app bundle is incomplete:\n\n" + "\n".join(f"- {m}" for m in missing))
            sys.exit()

    def set_in(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov")])
        if self.input_file: self.lbl_in.config(text=os.path.basename(self.input_file))

    def set_out(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir: self.lbl_out.config(text=self.output_dir)

    def start(self):
        if not self.input_file or not self.output_dir:
            return messagebox.showerror("Error", "Please complete Steps 1 and 2 first!")
        
        # MAC THREADING FIX: We must hide the Tkinter GUI to allow OpenCV to take over the main thread
        self.root.withdraw()
        self.root.after(100, self.launch_ai)

    def launch_ai(self):
        try:
            run_masker_engine(self.input_file, self.output_dir, self.mode.get())
            messagebox.showinfo("Success", "Process finished successfully!")
        except Exception as e:
            err = traceback.format_exc()
            print(err)
            messagebox.showerror("AI Engine Error", f"Details: {str(e)}")
        finally:
            self.root.deiconify() # Show the logo menu again

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()