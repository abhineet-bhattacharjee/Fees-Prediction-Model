import subprocess

schools = ['Business (MBA)', 'Law', 'Medical/Dental', 'Divinity']
scenarios = [
    {'inflation': None, 'endowment': None, 'name': 'baseline'},
    {'inflation': 3.5, 'endowment': 50.0, 'name': 'moderate'},
    {'inflation': 5.5, 'endowment': 60.0, 'name': 'high_growth'},
]