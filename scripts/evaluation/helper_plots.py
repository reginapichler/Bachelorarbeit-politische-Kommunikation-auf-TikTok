import matplotlib.pyplot as plt

# Original data
party = ["SPD", "CDU", "GRÜNE", "AfD", "CSU", "Die Linke", "SSW"]
seats = [120, 164, 85, 152, 44, 64, 1]
diff = [-86, 12, -33, 69, -1, 25, 0]
colors = [
    "#8B0000",  # SPD – dark red
    "#D3D3D3",  # CDU – light gray
    "#228B22",  # GRÜNE – green
    "#87CEFA",  # AfD – light blue
    "#00008B",  # CSU – dark blue
    "#800080",  # Die Linke – purple
    "#FFD700",  # SSW – yellow
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
    f"{p}\n{round(pc, 1)}% \n({'+' if d >= 0 else ''}{d} Sitze)"
    for p, pc, d in zip(party_sorted, perc_sorted, diff_sorted)
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
    labeldistance=1.1,
)
plt.title("Sitzverteilung im 21. Bundestag", fontsize=14)
plt.tight_layout()
plt.savefig("plots/sitzverteilung_bundestag.png", dpi=300)
plt.show()