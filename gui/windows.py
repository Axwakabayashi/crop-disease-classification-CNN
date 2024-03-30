import tkinter as tk
import tkinter.font as tk_font
from tkinter import ttk, filedialog, messagebox


class WelcomeWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Crop Savvier")
        self.window.geometry("700x400")
        self.logo = tk.PhotoImage(file="images/logo.png")
        self.logo = self.logo.subsample(2, 2)
        self.logo_label = tk.Label(self.window, image=self.logo)
        self.welcome_label = tk.Label(self.window, text="Welcome to Crop Savvier!",
                                      font=(tk_font.Font(size=30)))
        self.select_crop_label = tk.Label(self.window, text="Please Select Your Crop:")
        self.options = ["Cassava", "Corn", "Wheat"]
        self.selection = tk.StringVar()
        self.drop_down = ttk.Combobox(self.window, textvariable=self.selection,
                                      values=self.options)
        self.import_file_button = tk.Button(self.window, text="Import File",
                                            command=self.import_file)
        self.logo_label.pack()
        self.welcome_label.pack(pady=30)
        self.select_crop_label.pack()
        self.drop_down.pack(pady=10)
        self.import_file_button.pack()

    def import_file(self):
        if self.selection.get() == "":
            messagebox.showerror("Error", "No crop selected!")
            return
        path = filedialog.askopenfilename()
        print(path)

    def show(self):
        self.window.deiconify()

    def hide(self):
        self.window.deiconify()
