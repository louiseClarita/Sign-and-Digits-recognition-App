import sys
import tkinter as tk
import subprocess

import cv2  # version 3.2.0
def run_file():
    python_path = sys.executable  # Get the path of the current Python interpreter
    script_path = 'NEW_digit_recogV1.2.py'  # Path to the script you want to run

    subprocess.Popen([python_path, script_path])
def run_fileCamera():
    python_path = sys.executable  # Get the path of the current Python interpreter
    script_path = 'step_5_camera.py'  # Path to the script you want to run

    subprocess.Popen([python_path, script_path])
def run_checkAlpha():
    python_path = sys.executable  # Get the path of the current Python interpreter
    script_path = 'CheckALPHA.py'  # Path to the script you want to run

    subprocess.Popen([python_path, script_path])
def show_Info():
    root = tk.Tk()
    root.title("My App")

    # Configure the window's appearance
    root.geometry("1200x500")
    root.configure(bg="#f2f2f2")

    # Create a frame for the buttons
    button_frame = tk.Frame(root, bg="#f2f2f2")
    button_frame.pack(pady=20)

    # Create a label widget
    label = tk.Label(root, text="1 - Make sure you have a clean background, for accurate results!\n"
                                "2 - If needed check our manual picture to learn the signs to test them\n"
                                "3 - Press 'c' to capture screen, 'Q' to exit camera, 's' to capture and exit!\n"
                                "Have fun:")
    label.config(font=("Arial", 24))
    label.pack(padx=20, pady=20)  # Place the label in the window using pack layout

    # Create buttons and place them in the button_frame
    button2 = tk.Button(button_frame, text="Open Camera", command=run_fileCamera, padx=10, pady=5,
                        bg="#cc3333", fg="white", font=("Arial", 12))
    button2.pack(side=tk.LEFT, padx=10, pady=5)  # Place the button on the left side of the button_frame

    button1 = tk.Button(button_frame, text="With an old image", command=run_checkAlpha, padx=10, pady=5,
                        bg="#336699", fg="white", font=("Arial", 12))
    button1.pack(side=tk.LEFT, padx=10, pady=5)  # Place the button on the left side of the button_frame

    # Start the main event loop
    root.mainloop()


def exit_program():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("My App")

# Configure the window's appearance
root.geometry("300x200")
root.configure(bg="#f2f2f2")

# Create a frame for the buttons
button_frame = tk.Frame(root, bg="#f2f2f2")
button_frame.pack(pady=20)

# Create the buttons
button1 = tk.Button(button_frame, text="Predict Digits", command=run_file, padx=10, pady=5, bg="#336699", fg="white", font=("Arial", 12))
button1.pack(side=tk.TOP, padx=10)
button1 = tk.Button(button_frame, text="Predict Hand Signs", command=show_Info, padx=10, pady=5, bg="#336699", fg="white", font=("Arial", 12))
button1.pack(side=tk.TOP, padx=10)

button2 = tk.Button(button_frame, text="Exit", command=exit_program, padx=10, pady=5, bg="#cc3333", fg="white", font=("Arial", 12))
button2.pack(side=tk.BOTTOM, padx=10)

# Start the main event loop
root.mainloop()
