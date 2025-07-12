import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16
})

# Original data
party = ["CDU", "AfD", "SPD", "Bündnis 90/Die Grünen", "Die Linke", "CSU", "SSW"]
seats = [164, 151, 120, 85, 64, 44, 1]
diff = [12, 69, -86, -33, 25, -1, 0]
colors = [
    "#000000",  # CDU
    "#56B4E9",  # AfD
    "#D00000",  # SPD
    "#00B140",  # Grüne
    "#E10098",  # Die Linke
    "#00008B",  # CSU
    "#FFD700",  # SSW
]

# Calculate seat percentages
seats_total = sum(seats)
perc = [s / seats_total * 100 for s in seats]

# Combine all for sorting
combined = list(zip(seats, party, perc, diff, colors))
combined.sort(reverse=True)  # sort by seat count descending

# Unpack sorted values
seats_sorted, party_sorted, perc_sorted, diff_sorted, colors_sorted = zip(*combined)

# Create labels
labels = [
    "SSW" if p == "SSW" else f"{p}\n{round(pc, 1)}% \n({'+' if d >= 0 else ''}{d} Sitze)"
    for p, pc, d in zip(party, perc, diff)
]

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    seats_sorted,
    labels=labels,
    colors=colors_sorted,
    startangle=90,
    counterclock=False,
    wedgeprops={'edgecolor': 'black'},
    labeldistance=1.15,
)
plt.title("Sitzverteilung im 21. Bundestag", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/sitzverteilung_bundestag.png", dpi=300)
plt.show()