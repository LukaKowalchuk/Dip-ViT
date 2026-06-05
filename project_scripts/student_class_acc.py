import pandas as pd
from io import StringIO

# --- Original dataframe (Expert classifications) ---
expert_raw = """#\tExpert Classification\tNotes
1\t[Calliphoridae] Lucilia sericata and L. cuprina\t
2\t[Calliphoridae] Chrymsomya albiceps\t
3\t[Calliphoridae] Chrymsomya albiceps\t
4\t[Muscidae] Other\t
5\t[Calliphoridae] Chrysomya chloropyga\t
6\t[Calliphoridae] Lucilia sericata and L. cuprina\t
7\t[Muscidae] Synthesiomya nudiseta\tuncertain
8\t[Calliphoridae] Chrymsomya marginalis\t
9\t[Muscidae] Synthesiomya nudiseta\t
10\t[Calliphoridae] Chrymsomya marginalis\t
11\t[Calliphoridae] Chrymsomya marginalis\t
12\t[Muscidae] Other\t
13\t[Calliphoridae] Chrysomya chloropyga\t
14\t[Calliphoridae] Chrysomya chloropyga\t
15\tUnknown\t[Calliphoridae] Chrymsomya marginalis
16\t[Calliphoridae] Lucilia sericata and L. cuprina\t
17\t[Calliphoridae] Lucilia sericata and L. cuprina\t
18\t[Calliphoridae] Chrymsomya albiceps\t
19\t[Calliphoridae] Chrymsomya marginalis\t
20\t[Calliphoridae] Lucilia sericata and L. cuprina\t
21\t[Muscidae] Other\t
22\t[Muscidae] Other\t
23\t[Muscidae] Other\t
24\t[Calliphoridae] Chrymsomya albiceps\t
25\t[Calliphoridae] Chrymsomya marginalis\t
26\t[Calliphoridae] Lucilia sericata and L. cuprina\t
27\t[Calliphoridae] Chrymsomya marginalis\t
28\t[Muscidae] Other\t
29\t[Calliphoridae] Chrymsomya albiceps\t
30\t[Muscidae] Other\t
31\t[Muscidae] Other\t
32\t[Calliphoridae] Lucilia sericata and L. cuprina\t
33\t[Calliphoridae] Chrysomya chloropyga\t
34\t[Calliphoridae] Chrysomya chloropyga\t
35\t[Calliphoridae] Chrymsomya marginalis\t?
36\t[Calliphoridae] Lucilia sericata and L. cuprina\t
37\t[Calliphoridae] Chrymsomya albiceps\t
38\t[Calliphoridae] Chrysomya chloropyga\t
39\t[Calliphoridae] Chrymsomya albiceps\t
40\t[Muscidae] Other\t
41\t[Calliphoridae] Lucilia sericata and L. cuprina\t
42\t[Muscidae] Other\t
43\t[Calliphoridae] Lucilia sericata and L. cuprina\t
44\t[Calliphoridae] Lucilia sericata and L. cuprina\t
45\t[Muscidae] Other\t
46\t[Calliphoridae] Chrysomya chloropyga\t
47\t[Calliphoridae] Chrysomya chloropyga\t
48\t[Calliphoridae] Chrysomya chloropyga\t
49\t[Muscidae] Other\t
50\t[Calliphoridae] Lucilia sericata and L. cuprina\t
51\t[Calliphoridae] Chrymsomya albiceps\t?
52\t[Calliphoridae] Lucilia sericata and L. cuprina\t
53\t[Muscidae] Other\t
54\t[Calliphoridae] Chrysomya chloropyga\t
55\t[Muscidae] Other\t
56\t[Calliphoridae] Lucilia sericata and L. cuprina\t
57\t[Calliphoridae] Chrysomya chloropyga\t
58\t[Calliphoridae] Chrymsomya marginalis\t
59\t[Muscidae] Other\t
60\t[Calliphoridae] Chrysomya chloropyga\t
61\t[Muscidae] Other\t
62\t[Muscidae] Other\t
63\t[Muscidae] Other\t
64\tNone\tChrysomya putoria?
65\t[Calliphoridae] Chrymsomya marginalis\t
66\t[Muscidae] Other\t
67\t[Calliphoridae] Chrymsomya marginalis\t
68\t[Calliphoridae] Lucilia sericata and L. cuprina\t
69\t[Calliphoridae] Chrymsomya albiceps\t
70\t[Muscidae] Other\t
71\t[Muscidae] Other\t
72\t[Calliphoridae] Chrymsomya albiceps\t
73\t[Calliphoridae] Chrysomya chloropyga\t
74\t[Calliphoridae] Chrymsomya albiceps\t
75\t[Calliphoridae] Chrysomya chloropyga\t
76\t[Calliphoridae] Chrysomya chloropyga\t
77\t[Muscidae] Other\t
78\t[Calliphoridae] Chrymsomya marginalis\t
79\tUnknown\t[Calliphoridae] Chrymsomya albiceps / putoria
80\t[Calliphoridae] Chrymsomya marginalis\t
81\t[Calliphoridae] Chrysomya chloropyga\t
82\t[Muscidae] Other\t
83\t[Muscidae] Other\t
84\t[Calliphoridae] Chrymsomya albiceps\t
85\t[Calliphoridae] Chrymsomya albiceps\t
86\t[Muscidae] Other\t
87\t[Calliphoridae] Chrysomya chloropyga\t
88\t[Calliphoridae] Chrymsomya marginalis\t
89\t[Muscidae] Other\t
90\t[Calliphoridae] Chrymsomya marginalis\t
91\t[Calliphoridae] Chrymsomya albiceps\t
92\t[Muscidae] Other\t
93\t[Muscidae] Other\t
94\t[Muscidae] Other\t
95\t[Muscidae] Other\t
96\t[Calliphoridae] Lucilia sericata and L. cuprina\t
97\t[Calliphoridae] Chrymsomya albiceps\t
98\t[Muscidae] Other\t
99\t[Muscidae] Other\t
100\t[Calliphoridae] Chrymsomya marginalis\t
"""

df_expert = pd.read_csv(StringIO(expert_raw), sep="\t")
df_expert.columns = ["Sample", "Expert", "Notes"]

# For rows where Expert is "Unknown" or "None", use the Notes column as expert label
df_expert["Expert"] = df_expert.apply(
    lambda r: r["Notes"] if str(r["Expert"]).strip() in ["Unknown", "None"] else r["Expert"],
    axis=1
)

# --- Second dataframe (Student classifications) ---
student_data = {
    "Sample": list(range(1, 101)),
    "Student": [
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Muscidae] Other",
        "[Muscidae] Other",
        "[Muscidae] Other",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Muscidae] Other",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Other",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Muscidae] Other",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Muscidae] Other",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Muscidae] Other",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Muscidae] Other",
        "[Muscidae] Synthesiomya nudiseta",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Muscidae] Other",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Other",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Other",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Muscidae] Other",
        "[Muscidae] Other",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrysomya chloropyga",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrymsomya marginalis",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Synthesiomya nudiseta",
        "[Muscidae] Other",
        "[Muscidae] Other",
        "[Muscidae] Other",
        "[Calliphoridae] Lucilia sericata and L. cuprina",
        "[Calliphoridae] Chrymsomya albiceps",
        "[Muscidae] Synthesiomya nudiseta",
        "[Muscidae] Synthesiomya nudiseta",
        "[Calliphoridae] Chrymsomya marginalis",
    ],
    "FileName": [
        "A1.4","A1.5","C5#52","IMG_2180","C3#7","A11.1","IMG_1714","C9#54","IMG_2186","C13#71",
        "C14#89","IMG_2162","C4#9","C5#10","C16#93","A1.10","C2#2B","C6#23","C19#141","C2#31",
        "M72#91","M75#117","M76#117","C9#40","C21#104","C1#8","C23#103","M77#117","C37#36","M86#122",
        "IMG_2168","C1#9","C6#12","C7#14","C27#100","C2#1","C1#1","C8#15","C11#86","IMG_2174",
        "C1#7","M88#126","C1#74","C1#72","M89#109","C9#21","C10#20","C11#64","IMG_1921","C1#73",
        "C3#98","C1#5","IMG_2156","C1#3","IMG_1903","C1#40","C3#7","C8#22","M90#109","C1#75",
        "M56#99","IMG_1909","IMG_1848","C1#72","C10#31","B4.8","C7#18","L2#37","C1#99","B4.9",
        "IMG_1861","C13#88","C1#70","C16#130","C1#92","C17#132","B4.4","C5#74","C3#100","C6#39",
        "C2#5","B4.5","B4.1","C18#107","C22#98","IMG_1836","C1#3","C4#11","IMG_1818","C3#6",
        "C4#30","IMG_2192","B4.10","M71#91","M70#91","L1#9","C3#8","IMG_1757","IMG_1813","C2#5",
    ]
}

df_student = pd.DataFrame(student_data)

# --- Merge: fill Expert column from df_expert ---
df_merged = df_student.copy()
df_merged["Expert"] = df_expert["Expert"].values

# --- Flag differences ---
df_merged["Mismatch"] = df_merged["Student"] != df_merged["Expert"]

# Reorder columns
df_merged = df_merged[["Sample", "Student", "Expert", "Mismatch", "FileName"]]

# --- Overall stats ---
total = len(df_merged)
total_mismatches = df_merged["Mismatch"].sum()
overall_pct = total_mismatches / total * 100

print("=" * 70)
print(f"OVERALL: {total_mismatches} mismatches out of {total} → {overall_pct:.1f}% error rate")
print("=" * 70)

# --- Per-species breakdown (based on Student label) ---
print("\nPer-species mismatch breakdown (based on Student classification):\n")
species_stats = (
    df_merged.groupby("Student")["Mismatch"]
    .agg(Total="count", Mismatches="sum")
    .assign(Error_Pct=lambda x: (x["Mismatches"] / x["Total"] * 100).round(1))
    .reset_index()
    .rename(columns={"Student": "Species"})
)
print(species_stats.to_string(index=False))

# --- Show flagged rows ---
print("\n\nFlagged mismatches (Student ≠ Expert):\n")
flagged = df_merged[df_merged["Mismatch"]][["Sample", "Student", "Expert", "FileName"]]
print(flagged.to_string(index=False))
