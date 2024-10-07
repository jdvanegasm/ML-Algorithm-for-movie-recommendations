import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Combobox, Style
import joblib
import pandas as pd
from recommender import train_model

# Load the data and cosine similarity matrix
data_cleaned = pd.read_csv('../data/processed/movies_cleaned.csv')
cosine_sim = joblib.load('../neural_network/models/cosine_sim.pkl')

# Create the main application window
root = tk.Tk()
root.title("Movie Recommender")
root.geometry("500x300")  # Set the window size
root.config(bg="#f0f0f0")  # Set a light background color

# Set a style for the widgets
style = Style()
style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0", foreground="#333")
style.configure("TButton", font=("Helvetica", 10), background="#4CAF50", foreground="#fff")
style.configure("TCombobox", font=("Helvetica", 12))

# Create a label
label = tk.Label(root, text="Select a Movie:", font=("Helvetica", 14, "bold"), bg="#f0f0f0", fg="#333")
label.pack(pady=20)

# Create a Combobox with the list of movie titles
movie_titles = data_cleaned['title'].tolist()
combo = Combobox(root, values=movie_titles, width=50, font=("Helvetica", 12))
combo.pack(pady=10)

# Function to be called when the button is clicked
def on_recommend():
    selected_movie = combo.get()
    recommendations = train_model.get_recommendations(selected_movie, cosine_sim, data_cleaned)
    
    if isinstance(recommendations, str):
        # If no movie was found or an error occurred
        messagebox.showerror("Error", recommendations)
    else:
        # Display the recommendations in a messagebox
        messagebox.showinfo("Recommendations", "\n".join(recommendations))

# Create a button to trigger the recommendation
btn = tk.Button(root, text="Get Recommendations", command=on_recommend, font=("Helvetica", 12), bg="#4CAF50", fg="#fff", activebackground="#45a049", padx=10, pady=5)
btn.pack(pady=30)

# Start the Tkinter event loop
root.mainloop()
