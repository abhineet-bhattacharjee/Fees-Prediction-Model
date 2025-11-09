import subprocess

schools = [
    'Business__MBA_',
    'Design',
    'Divinity',
    'Education',
    'GSAS',
    'Government',
    'Law',
    'Medical_Dental',
    'Public_Health__1-Year_MPH_'
]
scenarios = [
    {'inflation': None, 'endowment': None, 'name': 'baseline'},
    # {'inflation': 3.5, 'endowment': 50.0, 'name': 'moderate'},
    # {'inflation': 5.5, 'endowment': 60.0, 'name': 'high_growth'},
]

years = range(1985, 2018)

for school in schools:
    for year in years:
        for scenario in scenarios:
            cmd = [
                'python', 'model_development.py',
                '--predict',
                '--school', school,
                '--year', str(year),
            ]

            if scenario['inflation']:
                cmd.extend(['--inflation', str(scenario['inflation'])])
            if scenario['endowment']:
                cmd.extend(['--endowment', str(scenario['endowment'])])

            print(f"{school} - {scenario['name']} - {year}")
            subprocess.run(cmd, capture_output=False)

print("\n" + "=" * 70)
print("ALL PREDICTIONS COMPLETE!")
print("=" * 70)
