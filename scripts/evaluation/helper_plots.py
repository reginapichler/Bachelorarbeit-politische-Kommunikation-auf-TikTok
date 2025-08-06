import matplotlib.pyplot as plt
from graphviz import Digraph

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18
})

def create_election_result_plot():
    """Creates a plot with the results of the Bundestagswahl 2025"""
    # Data
    parties = ["CDU", "AfD", "SPD", "Grüne", "Linke", "CSU", "SSW", "Sonstige"]
    percentages = [22.6, 20.8, 16.4, 11.6, 8.8,6.0, 0.2, 13.7]           # Current election results
    delta = [+3.6, +10.4, -9.3, -3.1, +3.9, +0.8, +0.0]               # Change compared to previous election
    colors = [
        "#000000", "#56B4E9", "#D00000", "#00B140", 
        "#E10098", "#00008B", "#FFD700", "#C1C1C1"
    ]

    # Create figure and bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(parties, percentages, color=colors, edgecolor='black')

    # Add text above bars: percentage + delta
    for bar, pct, d in zip(bars, percentages, delta):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        # Percent value
        plt.text(x, y + 1.8, f"{pct:.1f}%", ha='center', va='bottom', fontsize=16, fontweight='bold')

        # Change from previous election
        plt.text(x, y + 0.5, f"{d:+.1f}%", ha='center', va='bottom', fontsize=16, color='gray')

    # Chart formatting
    plt.ylabel("Stimmenanteil (%)")
    plt.xlabel("Partei")
    plt.title("Wahlergebnisse der Bundestagswahl 2025 (Zweitstimme)", fontsize=18, fontweight='bold')
    plt.ylim(0, max(percentages) + 5)
    plt.tight_layout()
    plt.savefig("plots/selfmade_plots/wahlergebnis", bbox_inches="tight", dpi=300)

def main():
    create_election_result_plot()



if __name__ == "__main__":
    main()