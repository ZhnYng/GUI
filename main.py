import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.font as tkFont
import pandas as pd
from tkinter import filedialog
from tkinter import Scrollbar
import numpy as np
from datetime import datetime
import joblib
import os
import sys

# Load the pre-trained classifier model
def resourcePath(relativePath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        basePath = sys._MEIPASS
    except Exception:
        basePath = os.path.abspath(".")

    return os.path.join(basePath, relativePath)

filename = "final_classifier.joblib"
clf = joblib.load(resourcePath(filename))

def validate_data_input(x):
    """
    Validate input data for manual entry.

    Args:
    - x (str): Input data

    Returns:
    - bool: True if validation passes, False otherwise
    """
    if len(x) == 0:
        messagebox.showerror("error", "Cannot be empty")
        return False
    else:
        try:
            x = float(x)
            return True
        except Exception as ep:
            messagebox.showerror("error", "Can only be integer/float")
            return False


def predict():
    """
    Make predictions based on manual input.
    """
    # Check for missing values in manual input
    if not data_input["%s" % features[0]].get():
        messagebox.showerror("error", "Missing value")
        return False
    X = [data_input["%s" % features[0]].get()]
    for i in range(len(features) - 1):
        if not data_input["%s" % features[i + 1]].get():
            messagebox.showerror("error", "Missing value")
            return False
        X += [data_input["%s" % features[i + 1]].get()]
    X = pd.DataFrame([X], columns=features)

    # Make predictions using the classifier
    y_pred = clf.predict(X)
    if y_pred[0] == 0:
        y_pred = "Safe"
    else:
        y_pred = "Not safe"

    # Update result label
    result["text"] = y_pred
    if result["text"] == "Not safe":
        result.config(fg="#FF0000")
    else:
        result.config(fg="#32CD32")

    # Display prediction probability
    prob = np.max(clf.predict_proba(X), axis=1)
    probability["text"] = "{:.2%}".format(prob[0])


table = None
data = None


def item_selected(event):
    """
    Handle selection of items in the data table.

    Args:
    - event (Event): Selection event
    """
    global table
    global data
    # Retrieve selected item from the table
    for selected_item in table.selection():
        item = table.item(selected_item)
        record = item["values"]
        # Update manual input with selected data
        for key, entry in data_input.items():
            value = tk.StringVar()
            value.set(float(data.loc[record[0]][key]))
            entry["textvariable"] = value
    predict()


def upload_action():
    """
    Handle file upload and processing.
    """
    # Prompt user to select a CSV file
    filename = filedialog.askopenfilename()
    if filename.endswith(".csv"):
        direct_file = filename.split("/")[-1]
        csv_file.set(direct_file)
        global data
        # Read data from the selected CSV file
        data = pd.read_csv(filename)
        y_pred = clf.predict(data)
        global table
        # Clear existing table if present
        if table is not None:
            clear()

        # Setup table with scroll bar
        scroll = Scrollbar(table_frame)
        scroll.pack(side="right", fill="y")
        table = ttk.Treeview(table_frame, yscrollcommand=scroll.set)
        table.bind("<<TreeviewSelect>>", item_selected)
        table.pack()
        scroll.config(command=table.yview)

        columns = ["booking_id", "Label", *features]
        table["columns"] = ("booking_id", "Label", *features)
        table.column("#0", width=0, stretch="no")
        table.heading("#0", text="", anchor="center")

        # Format columns in the table
        for column in columns:
            table.column(column, anchor="center", width=75)
            table.heading(column, text=column, anchor="center")

        # Insert data into the table
        booking_id = []
        pred_labels = []
        for i in range(len(data.index.values)):
            label = y_pred[i]
            if label == 0:
                label = "Safe"
            else:
                label = "Not safe"
            table.insert(
                parent="",
                index="end",
                iid=i,
                text="",
                values=(data.index.values[i], label, *data.values[i]),
            )
            booking_id.append(data.index.values[i])
            pred_labels.append(label)

        # Add booking_id and prediction columns to the data
        data.insert(0, "booking_id", booking_id, True)
        data.insert(1, "prediction", pred_labels, True)
        
        # Create export button
        export_btn = tk.Button(
            export_frame,
            text="Export result",
            cursor="hand2",
            font=("Poppins", 9),
            width=12,
            background=primary_color,
            foreground="white",
            command=download,
        )
        uploadBtn.pack()
        export_btn.pack()
    else:
        messagebox.showinfo("message", "The file must be in .csv format!")


def download():
    """
    Export the results to a CSV file.
    """
    if isinstance(data, pd.DataFrame):
        now = datetime.now()
        a = now.strftime("%d-%m-%Y %H%M")
        # Save the results to a CSV file
        data.to_csv(f"safety_predictions{a}.csv", index=False)
        messagebox.showinfo("message", "Results have been exported!")
    else:
        messagebox.showinfo("message", "No result to be exported!")


def clear():
    """
    Clear the UI components and reset values.
    """
    result["text"] = ""
    probability["text"] = ""
    csv_file.set("Upload a CSV file")
    # Destroy widgets in the table frame
    for widget in table_frame.winfo_children():
        widget.destroy()
    # Destroy widgets in the export frame
    for widget in export_frame.winfo_children():
        widget.destroy()
    # Reset manual input values
    for entry in data_input.values():
        value = tk.StringVar()
        value.set("")
        entry["textvariable"] = value


# Setup window
window = tk.Tk()
window.title("GoBest Cab Drivers Safety Prediction")
window.attributes("-fullscreen", True)
window.resizable(False, False)
window.bind("<Escape>", lambda event: window.quit())

# Define the colors
primary_color = "#FF5841"
secondary_color = "#b33c2e"
tertiary_color = "#ffd5d0"
fourth_color = "#ffaaa0"
bg_color = "#ffffff"

# Setup window grid
for i in range(12):
    window.columnconfigure(i, weight=1, minsize=50)
window.rowconfigure(0, weight=1, minsize=10)
for i in range(1, 19):
    window.rowconfigure(i, weight=1, minsize=45)

# Main large header
frame = tk.Frame(master=window)
frame.grid(row=2, column=0, columnspan=12)
ttk.Label(frame, text="Go Best", font=("Poppins ExtraBold", 40)).pack(
    anchor="w", side="left"
)
ttk.Label(
    frame, text=" Cab ", font=("Poppins ExtraBold", 40), foreground=primary_color
).pack(anchor="e", side="left")
ttk.Label(frame, text="Analytics", font=("Poppins ExtraBold", 40)).pack(side="left")

features = [
    "accuracy_mean",
    "bearing_mean",
    "acceleration_y_mean",
    "acceleration_z_mean",
    "second_mean",
    "gyro_magnitude_mean",
    "roll_mean",
    "driver_age_mean",
    "car_age_mean",
    "driver_start_age_mean",
    "G-Force_mean",
]

# Secondary header
frame = tk.Frame(master=window)
frame.grid(row=4, column=2, columnspan=1)
ttk.Label(
    frame,
    text="Manual Input",
    font=("Poppins ExtraBold", 20),
).pack(anchor="w", side="left")

# Map all the input values together
data_input = {}
rows = 3
cols = 4
input_frame = tk.Frame(master=window)
input_frame.grid(row=4, column=2, columnspan=4, rowspan=7)

# Create input fields for manual entry
for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        if index < len(features):
            frame = tk.Frame(master=input_frame, height=20)
            frame.grid(row=i + 1, column=j + 1, padx=10, pady=10)
            data_input_label = tk.Label(
                master=frame, text=features[index], width=20, font=("Poppins light", 10)
            )
            data_input["%s" % features[index]] = tk.Entry(
                master=frame, validate="key", font=("Poppins light", 10)
            )
            data_input["%s" % features[index]]["validatecommand"] = (
                data_input["%s" % features[index]].register(validate_data_input),
                "%P",
            )
            data_input_label.pack()
            data_input["%s" % features[index]].pack()

# Probability and Prediction
frame = tk.Frame(master=window)
frame.grid(row=4, column=7, columnspan=4, rowspan=3)
prediction = tk.Label(frame, text=f"Prediction : ")
result = tk.Label(frame)
proba = tk.Label(frame, text=f"Probability : ")
probability = tk.Label(frame)
prediction.pack()
result.pack()
proba.pack()
probability.pack()
result.config(font=tkFont.Font(family="Poppins medium", size=20))
probability.config(font=tkFont.Font(family="Poppins medium", size=20))

# Prediction button
frame = tk.Frame(master=window)
frame.grid(row=7, column=8, columnspan=2)
submitBtn = tk.Button(
    frame,
    text="Predict",
    cursor="hand2",
    font=("Poppins", 10),
    width=12,
    background=primary_color,
    foreground="white",
    command=predict,
)
submitBtn.pack()

# Clear button
frame = tk.Frame(master=window)
frame.grid(row=8, column=8, columnspan=2, rowspan=2)
clear_btn = tk.Button(
    frame,
    text="Clear",
    cursor="hand2",
    font=("Poppins", 10),
    width=12,
    background=primary_color,
    foreground="white",
    command=clear,
)
clear_btn.pack()

# Table
ttk.Label(master=window, text="Batch Input", font=("Poppins ExtraBold", 20)).place(
    relx=0.16, rely=0.55
)
table_frame = tk.Frame(master=window)

# Initialize placeholder table on program start
table = ttk.Treeview(table_frame)
table.pack(side="left")
columns = ["booking_id", "Label", *features]
table["columns"] = ("booking_id", "Label", *features)
table.column("#0", width=0, stretch="no")
table.heading("#0", text="", anchor="center")
for column in columns:
    table.column(column, anchor="center", width=75)
    table.heading(column, text=column, anchor="center")

table_frame.grid(row=11, columnspan=12, rowspan=6)

# Upload button
frame = tk.Frame(master=window)
frame.grid(row=17, column=4, columnspan=5)
csv_file = tk.StringVar(frame, value="Upload a CSV file")
t = tk.Label(
    master=frame,
    textvariable=csv_file,
    font=tkFont.Font(family="Poppins medium", size=12),
)
t.pack(side="left", padx=14)
uploadBtn = tk.Button(
    frame,
    text="Upload",
    cursor="hand2",
    font=("Poppins", 9),
    width=8,
    background=primary_color,
    foreground="white",
    command=upload_action,
)
uploadBtn.pack()

# Export button
export_frame = tk.Frame(master=window)
export_frame.grid(row=17, column=7, columnspan=2)

window.mainloop()