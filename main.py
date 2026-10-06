import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from moviepy import VideoFileClip
import threading

class VideoCutterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("កម្មវិធីកាត់វីដេអូ (Video Splitter Tool)")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        # Variables
        self.video_path = tk.StringVar()
        self.num_segments = tk.StringVar(value="6")

        # UI Layout
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(expand=True, fill="both")

        # File Selection
        ttk.Label(main_frame, text="ជ្រើសរើសឯកសារវីដេអូ:").grid(row=0, column=0, sticky="w", pady=5)
        self.file_entry = ttk.Entry(main_frame, textvariable=self.video_path, width=40)
        self.file_entry.grid(row=1, column=0, padx=5, pady=5)
        self.browse_btn = ttk.Button(main_frame, text="ស្វែងរក", command=self.browse_file)
        self.browse_btn.grid(row=1, column=1, padx=5, pady=5)

        # Number of segments
        ttk.Label(main_frame, text="ចំនួនភាគដែលត្រូវកាត់:").grid(row=2, column=0, sticky="w", pady=15)
        self.segments_entry = ttk.Entry(main_frame, textvariable=self.num_segments, width=10)
        self.segments_entry.grid(row=3, column=0, sticky="w", padx=5, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=20)

        self.status_label = ttk.Label(main_frame, text="រួចរាល់")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)

        # Start Button
        self.start_btn = ttk.Button(main_frame, text="ចាប់ផ្ដើមកាត់", command=self.start_cutting_thread)
        self.start_btn.grid(row=6, column=0, columnspan=2, pady=10)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)

    def start_cutting_thread(self):
        # Run in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self.cut_video)
        thread.start()

    def cut_video(self):
        path = self.video_path.get()
        segments_str = self.num_segments.get()

        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        try:
            num_segments = int(segments_str)
            if num_segments <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for segments.")
            return

        try:
            self.start_btn.config(state="disabled")
            self.browse_btn.config(state="disabled")

            self.status_label.config(text="Loading video...")
            clip = VideoFileClip(path)
            duration = clip.duration
            segment_duration = duration / num_segments

            # Create output directory
            output_dir = os.path.splitext(path)[0] + "_segments"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = os.path.basename(path)

            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)

                self.status_label.config(text=f"Cutting segment {i+1}/{num_segments}...")

                # Extract subclip
                subclip = clip.subclip(start_time, end_time)
                output_filename = f"part_{i+1}_{filename}"
                output_path = os.path.join(output_dir, output_filename)

                # Write file
                subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

                # Update progress
                self.progress['value'] = ((i + 1) / num_segments) * 100
                self.root.update_idletasks()

            clip.close()
            self.status_label.config(text="Done!")
            messagebox.showinfo("Success", f"Video split into {num_segments} segments successfully!\nSaved in: {output_dir}")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            self.status_label.config(text="Error occurred")
        finally:
            self.start_btn.config(state="normal")
            self.browse_btn.config(state="normal")
            self.progress['value'] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCutterApp(root)
    root.mainloop()
