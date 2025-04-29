import tkinter as tk
from tkinter import messagebox


def save_feedback(rating):
    # Enregistrer le feedback dans un fichier texte
    with open("feedback.txt", "a") as f:
        f.write(f"Feedback: {rating} étoiles\n")
    # Afficher un message dans l'interface graphique
    messagebox.showinfo(
        "Merci", f"Votre feedback ({rating} étoiles) a été enregistré !"
    )
    # Afficher le feedback dans le terminal
    print(f"Vous avez donné {rating} étoiles.")


def feedback_window():
    # Créer la fenêtre principale
    window = tk.Tk()
    window.title("Feedback avec étoiles")
    window.geometry("400x300")
    window.configure(bg="#f0f0f0")  # Couleur de fond

    # Titre
    label = tk.Label(
        window, text="Donnez votre avis :", font=("Arial", 16), bg="#f0f0f0"
    )
    label.pack(pady=10)

    # Label pour afficher le feedback sélectionné
    selected_feedback = tk.StringVar()
    selected_feedback.set("Aucun feedback sélectionné.")
    feedback_label = tk.Label(
        window, textvariable=selected_feedback, font=("Arial", 12), bg="#f0f0f0"
    )
    feedback_label.pack(pady=10)

    # Boutons pour les étoiles
    def update_feedback(rating):
        selected_feedback.set(f"Vous avez sélectionné {rating} étoiles.")
        save_feedback(rating)

    for i in range(1, 6):
        button = tk.Button(
            window,
            text="★" * i,
            font=("Arial", 18),
            bg="#ffcc00",
            activebackground="#ffd700",
            command=lambda rating=i: update_feedback(rating),
        )
        button.pack(pady=5)

    # Bouton Annuler
    cancel_button = tk.Button(
        window,
        text="Annuler",
        font=("Arial", 12),
        bg="#ff6666",
        activebackground="#ff4d4d",
        command=window.destroy,
    )
    cancel_button.pack(pady=20)

    # Lancer la boucle principale
    window.mainloop()


# Tester l'interface
if __name__ == "__main__":
    feedback_window()
