import subprocess

schools = ['Business (MBA)', 'Law', 'Medical/Dental', 'Divinity']
scenarios = [
    {'inflation': None, 'endowment': None, 'name': 'baseline'},
    {'inflation': 3.5, 'endowment': 50.0, 'name': 'moderate'},
    {'inflation': 5.5, 'endowment': 60.0, 'name': 'high_growth'},
]

for school in schools:
    for scenario in scenarios:
        cmd = [
            'python', 'model_deployment.py',
            '--school', school,
            '--start', '2025',
            '--end', '2030'
        ]

        if scenario['inflation']:
            cmd.extend(['--inflation', str(scenario['inflation'])])
        if scenario['endowment']:
            cmd.extend(['--endowment', str(scenario['endowment'])])

        print(f"\n{'=' * 60}")
        print(f"Running: {school} - {scenario['name']} scenario")
        print('=' * 60)

        result = subprocess.run(cmd, capture_output=False)