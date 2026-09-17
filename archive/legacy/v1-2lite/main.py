import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os, sys, traceback
from engine import run_masker_engine

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Mk Masker Pro v1.2 Lite")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        # Health Check
        if not os.path.exists(get_resource_path("ffmpeg")):
            messagebox.showerror("Error", "FFmpeg binary missing from bundle!")
            sys.exit()

        # Branding
        try:
            img = Image.open(get_resource_path("logo.png")).resize((150, 150))
            self.logo_img = ImageTk.PhotoImage(img)
            tk.Label(root, image=self.logo_img).pack(pady=20)
        except:
            tk.Label(root, text="Mk", font=("Arial", 40, "bold"), fg="#2ecc71").pack(pady=20)

        tk.Label(root, text="Mk Masker Pro", font=("Arial", 28, "bold")).pack()
        tk.Label(root, text="Lite Radiating Engine v1.2", font=("Arial", 10), fg="gray").pack()

        self.input_file = ""
        self.output_dir = ""
        self.mode = tk.StringVar(value="prores")

        tk.Button(root, text="1. Select Input Video", width=30, command=self.set_in).pack(pady=15)
        self.lbl_in = tk.Label(root, text="No video selected", wraplength=450, fg="blue")
        self.lbl_in.pack()

        tk.Button(root, text="2. Select Output Folder", width=30, command=self.set_out).pack(pady=10)
        self.lbl_out = tk.Label(root, text="No folder selected", wraplength=450, fg="blue")
        self.lbl_out.pack()

        m_frame = tk.LabelFrame(root, text="3. Export Mode", padx=20, pady=10)
        m_frame.pack(pady=20)
        tk.Radiobutton(m_frame, text="ProRes 4444 (Alpha)", variable=self.mode, value="prores").pack(side="left", padx=10)
        tk.Radiobutton(m_frame, text="B&W Mask (Fusion)", variable=self.mode, value="bw").pack(side="left", padx=10)

        self.btn = tk.Button(root, text="LAUNCH MK SELECTOR", bg="#2ecc71", fg="black", 
                            font=("Arial", 16, "bold"), height=2, width=25, command=self.start)
        self.btn.pack(pady=10)

    def set_in(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov")])
        if self.input_file: self.lbl_in.config(text=os.path.basename(self.input_file))

    def set_out(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir: self.lbl_out.config(text=self.output_dir)

    def start(self):
        if not self.input_file or not self.output_dir:
            return messagebox.showerror("Error", "Please select paths first!")
        self.root.withdraw()
        self.root.after(100, self.launch_ai)

    def launch_ai(self):
        try:
            run_masker_engine(self.input_file, self.output_dir, self.mode.get())
            messagebox.showinfo("Success", "Mk Masker Process finished!")
        except Exception as e:
            messagebox.showerror("Engine Error", f"Details: {str(e)}")
            print(traceback.format_exc())
        finally:
            self.root.deiconify()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()