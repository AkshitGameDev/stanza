import tkinter as tk
from tkinter import scrolledtext
import stanza

# Download the English models if not already installed.
# This step only needs to be done once.
stanza.download('en')

class StanzaGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stanza Text Processor")

        # Label for instructions
        self.label = tk.Label(master, text="Enter text for processing:")
        self.label.pack(pady=5)

        # Text widget for user input
        self.input_text = tk.Text(master, height=10, width=50)
        self.input_text.pack(pady=5)

        # Process button to trigger text processing
        self.process_button = tk.Button(master, text="Process", command=self.process_text)
        self.process_button.pack(pady=5)

        # Label for output results
        self.output_label = tk.Label(master, text="Processed Output:")
        self.output_label.pack(pady=5)

        # ScrolledText widget for displaying processed text output
        self.output_text = scrolledtext.ScrolledText(master, height=10, width=50)
        self.output_text.pack(pady=5)

        # Initialize the Stanza pipeline (processing: tokenization, POS tagging, and lemmatization)
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma", verbose=False)

    def process_text(self):
        # Retrieve the text from the input box
        raw_text = self.input_text.get("1.0", tk.END).strip()
        if not raw_text:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Please enter some text.")
            return

        # Process the text with Stanza
        doc = self.nlp(raw_text)
        result = ""

        # Iterate over sentences and words, and construct the output
        for sent in doc.sentences:
            for word in sent.words:
                result += f"Word: {word.text}\tLemma: {word.lemma}\tPOS: {word.pos}\n"
            result += "\n"  # Add a newline between sentences

        # Clear previous output and insert the new result
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, result)

if __name__ == "__main__":
    # Create the main window and start the application.
    root = tk.Tk()
    gui_app = StanzaGUI(root)
    root.mainloop()
