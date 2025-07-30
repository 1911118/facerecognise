import os
import shutil
from pathlib import Path
import face_recognition
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
import cv2
import platform
import subprocess

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Find Person in Photos (Glasses & Age Support)")
        self.root.geometry("800x600")
        
        # Setup logging
        logging.basicConfig(filename='error_log.txt', level=logging.INFO, 
                           format='%(asctime)s - %(message)s')
        
        # Variables
        self.input_dir = tk.StringVar(value=os.getcwd())
        self.reference_images = []  # Store multiple reference images
        self.tolerance = tk.DoubleVar(value=0.65)
        self.photo_references = []
        
        # GUI Elements
        tk.Label(root, text="Input Directory (Local or Network):").pack(pady=5)
        tk.Entry(root, textvariable=self.input_dir, width=50).pack(pady=5)
        tk.Button(root, text="Browse Input", command=self.browse_input).pack(pady=5)
        
        tk.Label(root, text="Reference Images:").pack(pady=5)
        self.ref_listbox = tk.Listbox(root, width=50, height=3)
        self.ref_listbox.pack(pady=5)
        tk.Button(root, text="Add Reference Image", command=self.add_reference).pack(pady=5)
        
        self.reference_label = tk.Label(root, text="No reference image selected")
        self.reference_label.pack(pady=5)
        
        tk.Label(root, text="Tolerance (0.4-0.9, higher is looser):").pack(pady=5)
        self.tolerance_entry = tk.Entry(root, textvariable=self.tolerance, width=10)
        self.tolerance_entry.pack(pady=5)
        
        self.run_button = tk.Button(root, text="Find Matching Faces", command=self.run_processing)
        self.run_button.pack(pady=10)
        
        self.progress = ttk.Progressbar(root, mode='determinate', maximum=100)
        self.progress.pack(pady=10, fill='x', padx=20)
        
        self.status = tk.Label(root, text="Ready", wraplength=700)
        self.status.pack(pady=5)
        
        tk.Label(root, text="Matched Images (Right-click to open file location):").pack(pady=5)
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(pady=5, fill='both', expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, height=200)
        self.canvas.pack(side='left', fill='both', expand=True)
        
        self.scrollbar = tk.Scrollbar(self.canvas_frame, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side='right', fill='y')
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor='nw')
        
        self.image_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
    
    def add_reference(self):
        file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file and file not in self.reference_images:
            self.reference_images.append(file)
            self.ref_listbox.insert(tk.END, os.path.basename(file))
            try:
                img = Image.open(file)
                img = img.resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.reference_label.configure(image=photo, text="")
                self.reference_label.image = photo
                self.photo_references.append(photo)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load reference image: {e}")
                logging.info(f"Cannot load reference image {file}: {e}")
    
    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
            # Convert to grayscale for histogram equalization
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_eq = cv2.equalizeHist(img_gray)
            # Convert back to RGB
            img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
            # Resize
            img_rgb = cv2.resize(img_rgb, (500, 500), interpolation=cv2.INTER_LANCZOS4)
            return img_rgb
        except Exception as e:
            self.update_status(f"Error preprocessing {image_path}: {e}")
            logging.info(f"Preprocessing {image_path}: {e}")
            return None
    
    def load_images_from_directory(self, directory):
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_paths = []
        try:
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.lower().endswith(image_extensions):
                        image_paths.append(os.path.join(root, f))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Cannot access directory {directory}: {e}"))
            logging.info(f"Accessing directory {directory}: {e}")
            return []
        return image_paths
    
    def get_face_encodings(self, image_path):
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return [], image_path
            # Use face landmarks for alignment
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                logging.info(f"No faces detected in {image_path}")
                return [], image_path
            encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=50)
            return encodings, image_path
        except Exception as e:
            self.update_status(f"Error processing {image_path}: {e}")
            logging.info(f"Processing {image_path}: {e}")
            return [], image_path
    
    def update_status(self, text):
        self.root.after(0, lambda: self.status.config(text=text))
    
    def update_progress(self, value):
        self.root.after(0, lambda: self.progress.config(value=value))
    
    def find_matching_faces(self, reference_paths, image_paths, tolerance):
        ref_encodings = []
        for ref_path in reference_paths:
            encodings, _ = self.get_face_encodings(ref_path)
            if encodings:
                ref_encodings.extend(encodings)
            else:
                logging.info(f"No faces in reference image {ref_path}")
        
        if not ref_encodings:
            self.root.after(0, lambda: messagebox.showerror("Error", "No faces detected in any reference image!"))
            return []
        
        # Average encodings for robustness
        ref_encoding = np.mean(ref_encodings, axis=0)
        matches = []
        
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            self.update_status(f"Processing image {i+1}/{total}: {os.path.basename(path)}")
            self.update_progress((i+1)/total*100)
            encodings, _ = self.get_face_encodings(path)
            for enc in encodings:
                distance = face_recognition.face_distance([ref_encoding], enc)[0]
                if distance <= tolerance:
                    matches.append(path)
                    logging.info(f"Match found for {path}, distance: {distance}")
                    break
                else:
                    logging.info(f"No match for {path}, distance: {distance}")
        
        return matches
    
    def download_image(self, image_path):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
            initialfile=os.path.basename(image_path)
        )
        if save_path:
            try:
                shutil.copy(image_path, save_path)
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Image saved to {save_path}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save image: {e}"))
                logging.info(f"Saving image {image_path} to {save_path}: {e}")
    
    def open_file_location(self, path):
        try:
            if platform.system() == "Windows":
                os.startfile(os.path.dirname(path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", os.path.dirname(path)])
            else:  # Linux and others
                subprocess.run(["xdg-open", os.path.dirname(path)])
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Cannot open file location: {e}"))
            logging.info(f"Opening file location {path}: {e}")
    
    def show_context_menu(self, event, path):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Open File Location", command=lambda: self.open_file_location(path))
        menu.post(event.x_root, event.y_root)
    
    def display_matches(self, matches):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        self.photo_references = []
        
        if not matches:
            tk.Label(self.image_frame, text="No matching images found").pack()
            return
        
        for i, path in enumerate(matches):
            try:
                img = Image.open(path)
                img = img.resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                frame = tk.Frame(self.image_frame)
                frame.grid(row=i, column=0, sticky='w', pady=5)
                
                label = tk.Label(frame, image=photo)
                label.image = photo
                self.photo_references.append(photo)
                label.pack(side='left')
                label.bind("<Button-3>", lambda e, p=path: self.show_context_menu(e, p))
                
                tk.Label(frame, text=f"Name: {os.path.basename(path)}", wraplength=200).pack(side='left', padx=10)
                tk.Label(frame, text=f"Path: {path}", wraplength=300).pack(side='left', padx=10)
                tk.Button(frame, text="Download", command=lambda p=path: self.download_image(p)).pack(side='left', padx=10)
            except Exception as e:
                self.update_status(f"Error displaying {path}: {e}")
                logging.info(f"Displaying {path}: {e}")
    
    def validate_tolerance(self):
        try:
            tolerance = float(self.tolerance.get())
            if not 0.4 <= tolerance <= 0.9:
                self.root.after(0, lambda: messagebox.showerror("Error", "Tolerance must be between 0.4 and 0.9"))
                return False
            return True
        except ValueError:
            self.root.after(0, lambda: messagebox.showerror("Error", "Invalid tolerance value"))
            return False
    
    def process_images(self):
        if not self.validate_tolerance():
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            self.root.after(0, lambda: self.progress.config(value=0))
            return
        
        input_dir = self.input_dir.get()
        reference_paths = self.reference_images
        tolerance = self.tolerance.get()
        
        if not os.path.exists(input_dir):
            self.root.after(0, lambda: messagebox.showerror("Error", "Input directory does not exist or is inaccessible!"))
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            self.root.after(0, lambda: self.progress.config(value=0))
            return
        
        if not reference_paths:
            self.root.after(0, lambda: messagebox.showerror("Error", "No reference images selected!"))
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            self.root.after(0, lambda: self.progress.config(value=0))
            return
        
        self.update_status("Scanning images...")
        image_paths = self.load_images_from_directory(input_dir)
        if not image_paths:
            self.root.after(0, lambda: messagebox.showerror("Error", "No images found in the input directory!"))
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            self.root.after(0, lambda: self.progress.config(value=0))
            return
        
        self.update_status(f"Found {len(image_paths)} images. Processing faces...")
        matches = self.find_matching_faces(reference_paths, image_paths, tolerance)
        
        self.update_status(f"Found {len(matches)} matching images.")
        self.root.after(0, lambda: self.display_matches(matches))
        
        with open("matches_log.txt", "w") as f:
            f.write(f"Matched Images:\n{', '.join(matches)}\n")
        
        self.root.after(0, lambda: messagebox.showinfo("Success", "Face recognition completed! Check error_log.txt for issues."))
        self.root.after(0, lambda: self.run_button.config(state='normal'))
        self.root.after(0, lambda: self.progress.config(value=0))
    
    def run_processing(self):
        self.run_button.config(state='disabled')
        self.progress.config(value=0)
        threading.Thread(target=self.process_images, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()