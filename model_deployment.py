import subprocess

schools = ['Business__MBA_', 'Law', 'Medical_Dental', 'Divinity']
scenarios = [
    {'inflation': None, 'endowment': None, 'name': 'baseline'},
    {'inflation': 3.5, 'endowment': 50.0, 'name': 'moderate'},
    {'inflation': 5.5, 'endowment': 60.0, 'name': 'high_growth'},
]

years = range(2017, 2031)  # 2017-2030

for year in years:
    print(f"\n{'#' * 70}")
    print(f"# YEAR {year}")
    print('#' * 70)

    for school in schools:
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

            print(f"\n{'=' * 60}")
            print(f"{school} - {scenario['name']}")
            print('=' * 60)

            subprocess.run(cmd, capture_output=False)

print("\n" + "=" * 70)
print("ALL PREDICTIONS COMPLETE!")
print("=" * 70)
